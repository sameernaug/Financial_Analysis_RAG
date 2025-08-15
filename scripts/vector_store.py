

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

class FinancialVectorStore:
    def __init__(self, persist_directory="./models/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize local embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collections for different data types
        self.collections = {
            'market_data': self.client.get_or_create_collection("market_data"),
            'news': self.client.get_or_create_collection("news"),
            'sec_filings': self.client.get_or_create_collection("sec_filings")
        }
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def prepare_chunks_for_embedding(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Prepare chunks for embedding based on type"""
        prepared = {
            'market_data': [],
            'news': [],
            'sec_filings': []
        }
        
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            
            if chunk_type in prepared:
                # Create embedding text
                if chunk_type == 'market_data':
                    text = f"Market data for {chunk.get('symbol', 'unknown')}: " \
                           f"Period {chunk.get('start_date', '')} to {chunk.get('end_date', '')}. " \
                           f"Average price ${chunk.get('metadata', {}).get('avg_price', 0):.2f}. " \
                           f"Trend: {chunk.get('metadata', {}).get('trend', 'unknown')}. " \
                           f"Volatility: {chunk.get('metadata', {}).get('volatility', 0):.4f}"
                
                elif chunk_type == 'news':
                    text = f"Financial news: {chunk.get('text', '')} " \
                           f"Published: {chunk.get('published', '')}. " \
                           f"Sentiment: {chunk.get('sentiment', 'neutral')}"
                
                elif chunk_type == 'sec_filing':
                    text = f"SEC filing for {chunk.get('company_name', 'unknown')}: " \
                           f"Type {chunk.get('filing_type', 'unknown')}. " \
                           f"Section: {chunk.get('section', 'unknown')}. " \
                           f"Content: {chunk.get('content', '')}"
                
                chunk['embedding_text'] = text
                prepared[chunk_type].append(chunk)
        
        return prepared
    
    def add_chunks_to_vector_store(self, chunks: List[Dict]):
        """Add chunks to appropriate collections in vector store"""
        prepared_chunks = self.prepare_chunks_for_embedding(chunks)
        
        for chunk_type, chunk_list in prepared_chunks.items():
            if not chunk_list:
                continue
            
            collection = self.collections[chunk_type]
            
            # Generate embeddings
            texts = [chunk['embedding_text'] for chunk in chunk_list]
            embeddings = self.generate_embeddings(texts)
            
            # Prepare metadata
            ids = [f"{chunk_type}_{i}" for i in range(len(chunk_list))]
            metadatas = []
            
            for chunk in chunk_list:
                # Get date from appropriate field based on chunk type
                date_value = chunk.get('start_date', '') or chunk.get('published', '') or chunk.get('filing_date', '')
                
                # Convert date to timestamp if it's a string
                try:
                    if date_value and isinstance(date_value, str):
                        date_timestamp = float(pd.to_datetime(date_value).timestamp())
                    elif date_value:
                        date_timestamp = float(date_value)
                    else:
                        # Default to current time if no date available
                        date_timestamp = float(datetime.now().timestamp())
                except Exception:
                    # If conversion fails, use current time
                    date_timestamp = float(datetime.now().timestamp())
                
                metadata = {
                    'type': chunk_type,
                    'symbol': str(chunk.get('symbol', '')).upper(),  # Normalize symbol to uppercase
                    'date': date_timestamp,  # Store as float timestamp
                    'sentiment': chunk.get('sentiment', ''),
                    'volatility': str(chunk.get('metadata', {}).get('volatility', '')),
                    'trend': chunk.get('metadata', {}).get('trend', '')
                }
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(chunk_list)} chunks to {chunk_type} collection")
    
    def query_vector_store(self, query: str, collection_name: str, n_results: int = 5):
        """Query the vector store for relevant chunks"""
        if collection_name not in self.collections:
            return []
        
        collection = self.collections[collection_name]
        query_embedding = self.generate_embeddings([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def query_all_collections(self, query: str, n_results: int = 3):
        """Query all collections and combine results"""
        all_results = {}
        
        for collection_name in self.collections.keys():
            results = self.query_vector_store(query, collection_name, n_results)
            all_results[collection_name] = results
        
        return all_results
    
    def get_collection_stats(self):
        """Get statistics for all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {'count': count}
            except Exception as e:
                stats[name] = {'count': 0, 'error': str(e)}
        
        return stats
    
    def similarity_search_with_temporal_filter(self, query: str, 
                                               start_date: str = None, 
                                               end_date: str = None,
                                               symbols: List[str] = None,
                                               n_results: int = 5):
        """Perform similarity search with temporal and symbol filters"""
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        all_results = []
        
        for collection_name, collection in self.collections.items():
            # Build filter conditions
            filter_conditions = []
            
            # Add date filter conditions - convert string dates to timestamps for ChromaDB
            if start_date:
                try:
                    # Convert date string to timestamp if it's a string
                    if isinstance(start_date, str):
                        # Convert to pandas datetime and then to timestamp (float)
                        start_timestamp = pd.to_datetime(start_date).timestamp()
                        filter_conditions.append({"date": {"$gte": float(start_timestamp)}})
                    else:
                        # If it's already a number, use it directly
                        filter_conditions.append({"date": {"$gte": float(start_date)}})
                except Exception:
                    # If conversion fails, skip this filter
                    pass
                    
            if end_date:
                try:
                    # Convert date string to timestamp if it's a string
                    if isinstance(end_date, str):
                        # Convert to pandas datetime and then to timestamp (float)
                        end_timestamp = pd.to_datetime(end_date).timestamp()
                        filter_conditions.append({"date": {"$lte": float(end_timestamp)}})
                    else:
                        # If it's already a number, use it directly
                        filter_conditions.append({"date": {"$lte": float(end_date)}})
                except Exception:
                    # If conversion fails, skip this filter
                    pass
            
            # Add symbol filter condition (case-insensitive)
            if symbols and len(symbols) > 0:
                # Normalize symbols to uppercase for consistency
                normalized_symbols = [symbol.upper() for symbol in symbols if symbol]
                
                # For a single symbol, use simple equality
                if len(normalized_symbols) == 1:
                    filter_conditions.append({"symbol": normalized_symbols[0]})
                # For multiple symbols, use $or operator with multiple equality conditions
                elif len(normalized_symbols) > 1:
                    symbol_conditions = [{"symbol": symbol} for symbol in normalized_symbols]
                    filter_conditions.append({"$or": symbol_conditions})
            
            # Combine filter conditions with $and operator if multiple conditions exist
            where_clause = None
            if filter_conditions:
                if len(filter_conditions) == 1:
                    where_clause = filter_conditions[0]
                else:
                    where_clause = {"$and": filter_conditions}
            
            # Query with filters
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            # Add collection info to results
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'collection': collection_name,
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    }
                    all_results.append(result)
        
        # Sort by distance (relevance)
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:n_results]