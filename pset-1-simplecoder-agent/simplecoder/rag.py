"""
SimpleCoder RAG - Retrieval-Augmented Generation for code search.

This module implements AST-based code chunking and embedding-based retrieval
for semantic code search across a codebase.
"""

import ast
import glob
from pathlib import Path
from typing import Any
import numpy as np
import litellm


class RAGSystem:
    # Indexes Python definitions (functions/classes) and retrieves the closest matches for a query.
    
    def __init__(
        self,
        embedder_model: str = "gemini/text-embedding-004",
        index_pattern: str = "**/*.py",
        base_dir: str = "."
    ):
        self.embedder_model = embedder_model
        self.index_pattern = index_pattern
        self.base_dir = Path(base_dir).resolve()
        
        # Storage for code chunks and embeddings
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
    
    def index_codebase(self) -> None:
        # Walk the repo, extract chunks from .py files, then embed them.
        # Find all Python files
        pattern = str(self.base_dir / self.index_pattern)
        python_files = glob.glob(pattern, recursive=True)
        
        # Extract code chunks from each file
        for filepath in python_files:
            try:
                self._index_file(filepath)
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        # Generate embeddings for all chunks
        if self.chunks:
            self._generate_embeddings()
    
    def _index_file(self, filepath: str) -> None:
        # Index a single Python file using AST.
        
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Skip files with syntax errors
            return
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function
                func_code = ast.get_source_segment(code, node)
                if func_code:
                    self.chunks.append({
                        "file": filepath,
                        "type": "function",
                        "name": node.name,
                        "code": func_code,
                        "lineno": node.lineno
                    })
            
            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_code = ast.get_source_segment(code, node)
                if class_code:
                    self.chunks.append({
                        "file": filepath,
                        "type": "class",
                        "name": node.name,
                        "code": class_code,
                        "lineno": node.lineno
                    })
    
    def _generate_embeddings(self) -> None:
        # Generate embeddings for all code chunks.
        
        # Prepare texts for embedding
        texts = []
        for chunk in self.chunks:
            # Create a rich text representation
            text = f"{chunk['type']}: {chunk['name']}\n{chunk['code']}"
            texts.append(text)
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = litellm.embedding(
                    model=self.embedder_model,
                    input=batch
                )
                
                batch_embeddings = [item['embedding'] for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            except Exception as e:
                # If embedding fails, use zero vectors
                embedding_dim = 768  # Default dimension
                all_embeddings.extend([
                    [0.0] * embedding_dim for _ in batch
                ])
        
        self.embeddings = np.array(all_embeddings)
    
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        # Search for code chunks semantically similar to the query.
        
        if not self.chunks or self.embeddings is None:
            return []
        
        # Generate query embedding
        try:
            response = litellm.embedding(
                model=self.embedder_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0]['embedding'])
        except Exception:
            return []
        
        # Compute cosine similarities
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        # Get statistics about the indexed codebase.
        
        if not self.chunks:
            return {
                "total_chunks": 0,
                "total_functions": 0,
                "total_classes": 0,
                "total_files": 0
            }
        
        files = set(chunk["file"] for chunk in self.chunks)
        functions = sum(1 for chunk in self.chunks if chunk["type"] == "function")
        classes = sum(1 for chunk in self.chunks if chunk["type"] == "class")
        
        return {
            "total_chunks": len(self.chunks),
            "total_functions": functions,
            "total_classes": classes,
            "total_files": len(files)
        }