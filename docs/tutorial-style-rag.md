# Tutorial-Style RAG with RWSVectorStore

This document shows how to use our ai-tools services in the same way as the LangChain tutorial, but with RWSVectorStore as the backend.

## Quick Start

```typescript
import { LangChainEmbeddingService } from '@rws-framework/ai-tools';
import { Document } from '@langchain/core/documents';

// Initialize embedding service
const embeddingService = new LangChainEmbeddingService();
await embeddingService.initialize({
    provider: 'cohere',
    apiKey: process.env.COHERE_API_KEY,
    model: 'embed-v4.0'
});

// Create documents (like tutorial's document loading)
const documents = [
    new Document({
        pageContent: "Task decomposition breaks complex tasks into steps.",
        metadata: { source: 'tutorial' }
    }),
    // ... more documents
];

// Create vector store (tutorial-style)
const vectorStore = await embeddingService.createVectorStore(documents);

// Similarity search (tutorial-style)
const results = await embeddingService.similaritySearch(vectorStore, "What is task decomposition?", 3);
```

## Comparison with LangChain Tutorial

### LangChain Tutorial Approach:
```typescript
// Tutorial code
const vectorStore = new MemoryVectorStore(embeddings);
await vectorStore.addDocuments(documents);
const results = await vectorStore.similaritySearch(query, k);
```

### Our AI-Tools Approach:
```typescript
// Our equivalent code using RWSVectorStore
const vectorStore = await embeddingService.createVectorStore(documents);
const results = await embeddingService.similaritySearch(vectorStore, query, k);
```

## Available Methods

### 1. Simple Similarity Search
```typescript
const docs = await embeddingService.similaritySearch(vectorStore, query, k);
// Returns: Document[]
```

### 2. Similarity Search with Scores
```typescript
const results = await embeddingService.similaritySearchWithScore(vectorStore, query, k);
// Returns: [Document, number][]
```

### 3. Enhanced Search with Filters
```typescript
const results = await vectorStoreService.searchSimilar({
    query: "your query",
    maxResults: 5,
    similarityThreshold: 0.1,
    filter: {
        knowledgeIds: ['28'],
        documentIds: ['doc1', 'doc2']
    }
});
// Returns: IVectorSearchResponse
```

## Integration with RAG Module

The RAG module already uses these services:

```typescript
// In backend/src/app/rag_module/rag.service.ts
constructor(
    private embeddingService: LangChainEmbeddingService,
    private langChainRAGService: LangChainRAGService
) {}
```

## Benefits of Our Approach

1. **Tutorial Compatibility**: Same interface as LangChain tutorial
2. **RWSVectorStore Backend**: Uses our proven vector storage system  
3. **Memory & FAISS Support**: Can use both in-memory and persistent storage
4. **Knowledge Filtering**: Built-in support for multi-tenant scenarios
5. **Polish Content Optimized**: Similarity thresholds tuned for non-English content

## Configuration

```typescript
// Centralized configuration (from RAG service)
static EMBEDDING_CONFIG = {
    provider: 'cohere',
    model: 'embed-v4.0',
    batchSize: 96
};

static RAG_CONFIG = {
    search: {
        defaultSimilarityThreshold: 0.1,  // Optimized for Polish content
        maxResults: 5
    },
    vectorStore: {
        type: 'memory',  // or 'faiss' for persistence
        autoSave: true
    }
};
```

## Examples

See `/examples/tutorial-style-rag.ts` for complete working examples.
