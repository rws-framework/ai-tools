/**
 * Example: LangChain Tutorial-Style RAG with RWSVectorStore
 * 
 * This demonstrates how to use our ai-tools services in the same way
 * as the LangChain tutorial, but with RWSVectorStore backend.
 */

import { LangChainEmbeddingService } from '../src/services/LangChainEmbeddingService';
import { LangChainVectorStoreService } from '../src/services/LangChainVectorStoreService';
import { Document } from '@langchain/core/documents';

async function tutorialStyleRAGExample() {
    // Initialize services like the tutorial
    const embeddingService = new LangChainEmbeddingService();
    const vectorStoreService = new LangChainVectorStoreService();

    // Configure embeddings (Cohere instead of OpenAI)
    await embeddingService.initialize({
        provider: 'cohere',
        apiKey: process.env.COHERE_API_KEY || '',
        model: 'embed-v4.0',
        batchSize: 96
    });

    // Initialize vector store service
    await vectorStoreService.initialize(embeddingService, {
        type: 'memory',
        similarityThreshold: 0.1,
        maxResults: 5
    });

    // Sample documents (like the tutorial's blog post chunks)
    const sampleTexts = [
        "Task decomposition is the process of breaking down complex tasks into smaller, more manageable steps.",
        "Chain of Thought (CoT) prompting helps models think step by step to solve complex problems.",
        "Tree of Thoughts extends CoT by exploring multiple reasoning possibilities at each step.",
        "RAG (Retrieval Augmented Generation) combines retrieval and generation for better answers.",
        "Vector databases store embeddings to enable semantic search over documents."
    ];

    // Create documents from texts (like tutorial's document loading)
    const documents: Document[] = [];
    for (let i = 0; i < sampleTexts.length; i++) {
        documents.push(new Document({
            pageContent: sampleTexts[i],
            metadata: {
                id: `doc_${i}`,
                source: 'tutorial_example',
                chunkIndex: i
            }
        }));
    }

    // Add documents to vector store (like tutorial's vectorStore.addDocuments)
    await vectorStoreService.addDocuments(documents);

    // Create a vector store for similarity search (tutorial-style)
    const vectorStore = await embeddingService.createVectorStore(documents, { type: 'memory' });

    // Perform similarity search like the tutorial
    const query = "What is task decomposition?";
    console.log(`\n🔍 Searching for: "${query}"`);

    // Method 1: Tutorial-style similarity search (returns documents only)
    const similarDocs = await embeddingService.similaritySearch(vectorStore, query, 3);
    console.log('\n📄 Similar documents (tutorial-style):');
    similarDocs.forEach((doc, index) => {
        console.log(`${index + 1}. ${doc.pageContent}`);
    });

    // Method 2: Similarity search with scores (tutorial-style)
    const similarDocsWithScores = await embeddingService.similaritySearchWithScore(vectorStore, query, 3);
    console.log('\n📊 Similar documents with scores:');
    similarDocsWithScores.forEach(([doc, score], index) => {
        console.log(`${index + 1}. Score: ${score.toFixed(4)} - ${doc.pageContent}`);
    });

    // Method 3: Using the vector store service (our enhanced approach)
    const searchResults = await vectorStoreService.searchSimilar({
        query,
        maxResults: 3,
        similarityThreshold: 0.1
    });

    console.log('\n🎯 Enhanced search results:');
    searchResults.results.forEach((result, index) => {
        console.log(`${index + 1}. Score: ${result.score.toFixed(4)} - ${result.content}`);
        console.log(`   Chunk ID: ${result.chunkId}`);
    });

    return {
        tutorialStyle: similarDocs,
        withScores: similarDocsWithScores,
        enhanced: searchResults.results
    };
}

// Example usage with knowledge filtering (like our current RAG system)
async function knowledgeFilteredExample() {
    const embeddingService = new LangChainEmbeddingService();
    const vectorStoreService = new LangChainVectorStoreService();

    await embeddingService.initialize({
        provider: 'cohere',
        apiKey: process.env.COHERE_API_KEY || '',
        model: 'embed-v4.0'
    });

    await vectorStoreService.initialize(embeddingService, {
        type: 'memory',
        similarityThreshold: 0.1
    });

    // Documents with knowledge IDs (like our current system)
    const documents = [
        new Document({
            pageContent: "Testing prototypes is crucial for product development",
            metadata: { knowledgeId: '28', documentId: 'test_doc', chunkIndex: 0 }
        }),
        new Document({
            pageContent: "Quality assurance ensures product reliability",
            metadata: { knowledgeId: '28', documentId: 'test_doc', chunkIndex: 1 }
        }),
        new Document({
            pageContent: "User feedback drives iterative improvements",
            metadata: { knowledgeId: '29', documentId: 'feedback_doc', chunkIndex: 0 }
        })
    ];

    await vectorStoreService.addDocuments(documents);

    // Search with knowledge filtering (like our RAG system does)
    const results = await vectorStoreService.searchSimilar({
        query: "opisz założenia dokumentu",
        maxResults: 2,
        similarityThreshold: 0.1,
        filter: {
            knowledgeIds: ['28']  // Only search in knowledge 28
        }
    });

    console.log('\n🔍 Knowledge-filtered search results:');
    results.results.forEach((result, index) => {
        console.log(`${index + 1}. Score: ${result.score.toFixed(4)}`);
        console.log(`   Content: ${result.content}`);
        console.log(`   Knowledge ID: ${result.metadata.knowledgeId}`);
    });

    return results;
}

// Export for use in tests or other modules
export { tutorialStyleRAGExample, knowledgeFilteredExample };
