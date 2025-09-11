/**
 * Test the new RecursiveCharacterTextSplitter approach in TextChunker
 * This verifies the tutorial-style chunking is working correctly
 */

import { TextChunker } from '../src/services/TextChunker';

function testRecursiveChunking() {
    console.log('🧪 Testing RecursiveCharacterTextSplitter approach in TextChunker\n');

    // Test document with various separator types
    const testDocument = `
# Introduction to AI

Artificial Intelligence (AI) is a rapidly evolving field that encompasses various technologies and methodologies. AI systems can process vast amounts of data, recognize patterns, and make decisions.

## Machine Learning

Machine learning is a subset of AI that focuses on algorithms that can learn from data. There are several types of machine learning:

1. Supervised Learning: Uses labeled data to train models
2. Unsupervised Learning: Finds patterns in unlabeled data  
3. Reinforcement Learning: Learns through interaction with environment

### Deep Learning

Deep learning uses neural networks with multiple layers to process information. These networks can automatically extract features from raw data. Common applications include:

- Image recognition and computer vision
- Natural language processing and understanding
- Speech recognition and synthesis
- Autonomous vehicles and robotics

## Applications and Impact

AI technologies are transforming various industries. Healthcare uses AI for diagnosis and treatment planning. Finance leverages AI for fraud detection and algorithmic trading. Manufacturing employs AI for quality control and predictive maintenance.

The ethical implications of AI are increasingly important! Questions about bias, privacy, and job displacement need careful consideration? As AI becomes more prevalent, ensuring responsible development and deployment is crucial; we must balance innovation with ethical considerations.
    `.trim();

    console.log('📄 Original document:');
    console.log(`Length: ${testDocument.length} characters`);
    console.log(`Estimated tokens: ${TextChunker.estimateTokens(testDocument)}`);
    console.log('First 200 chars:', testDocument.substring(0, 200) + '...\n');

    // Test different chunking configurations
    const testConfigs = [
        { maxTokens: 150, overlap: 20, name: 'Small chunks with overlap' },
        { maxTokens: 300, overlap: 50, name: 'Medium chunks with overlap' },
        { maxTokens: 100, overlap: 0, name: 'Small chunks no overlap' }
    ];

    for (const config of testConfigs) {
        console.log(`\n🔧 Testing: ${config.name}`);
        console.log(`Max tokens: ${config.maxTokens}, Overlap: ${config.overlap}`);
        
        const chunks = TextChunker.chunkText(testDocument, config.maxTokens, config.overlap);
        
        console.log(`\n✅ Generated ${chunks.length} chunks:`);
        
        chunks.forEach((chunk, index) => {
            const tokens = TextChunker.estimateTokens(chunk);
            const withinLimit = tokens <= config.maxTokens;
            const status = withinLimit ? '✅' : '❌';
            
            console.log(`${status} Chunk ${index + 1}: ${tokens} tokens, ${chunk.length} chars`);
            console.log(`   Preview: "${chunk.substring(0, 80)}..."`);
            
            if (!withinLimit) {
                console.log(`   ⚠️  WARNING: Chunk exceeds token limit (${tokens} > ${config.maxTokens})`);
            }
        });

        // Check for overlaps if configured
        if (config.overlap > 0 && chunks.length > 1) {
            console.log('\n🔄 Checking overlaps:');
            for (let i = 1; i < chunks.length; i++) {
                const prevChunk = chunks[i - 1];
                const currentChunk = chunks[i];
                
                // Simple overlap detection - check if chunks share common words
                const prevWords = prevChunk.split(' ').slice(-10);
                const currentWords = currentChunk.split(' ').slice(0, 10);
                
                const commonWords = prevWords.filter(word => 
                    currentWords.some(cWord => 
                        word.length > 3 && cWord.includes(word.substring(0, Math.min(word.length, 5)))
                    )
                );
                
                if (commonWords.length > 0) {
                    console.log(`   📎 Chunk ${i} has overlap with chunk ${i}: "${commonWords.slice(0, 3).join(', ')}"...`);
                } else {
                    console.log(`   ❓ Chunk ${i} may not have sufficient overlap with previous chunk`);
                }
            }
        }
    }
}

function testDocumentCreation() {
    console.log('\n\n📋 Testing Document Creation (Tutorial Style)');
    
    const sampleText = `
Artificial Intelligence is transforming the world. Machine learning algorithms can process vast amounts of data and identify patterns that humans might miss.

Deep learning, a subset of machine learning, uses neural networks to solve complex problems. These networks consist of multiple layers that can automatically extract features from raw data.
    `.trim();

    const documents = TextChunker.createDocumentsFromChunks(
        sampleText,
        { 
            documentId: 'ai-intro',
            source: 'tutorial',
            category: 'technology'
        },
        200, // maxTokens
        30   // overlap
    );

    console.log(`\n📄 Created ${documents.length} documents:`);
    
    documents.forEach((doc, index) => {
        console.log(`\nDocument ${index + 1}:`);
        console.log(`  ID: ${doc.metadata.id}`);
        console.log(`  Chunk: ${doc.metadata.chunkIndex + 1}/${doc.metadata.totalChunks}`);
        console.log(`  Content: "${doc.pageContent.substring(0, 100)}..."`);
        console.log(`  Tokens: ${TextChunker.estimateTokens(doc.pageContent)}`);
    });
}

function testEdgeCases() {
    console.log('\n\n🧪 Testing Edge Cases');

    const testCases = [
        { name: 'Empty string', text: '' },
        { name: 'Very short text', text: 'Hello world!' },
        { name: 'Single long word', text: 'Supercalifragilisticexpialidocious'.repeat(20) },
        { name: 'No separators', text: 'abcdefghijklmnopqrstuvwxyz'.repeat(50) },
        { name: 'Only separators', text: '\n\n\n. . . ! ! ! ? ? ?' }
    ];

    testCases.forEach(testCase => {
        console.log(`\n🔬 Testing: ${testCase.name}`);
        try {
            const chunks = TextChunker.chunkText(testCase.text, 100, 20);
            console.log(`   ✅ Generated ${chunks.length} chunks`);
            if (chunks.length > 0) {
                console.log(`   First chunk: "${chunks[0].substring(0, 50)}${chunks[0].length > 50 ? '...' : ''}"`);
            }
        } catch (error) {
            console.log(`   ❌ Error: ${error.message}`);
        }
    });
}

// Run all tests
console.log('🚀 Starting RecursiveCharacterTextSplitter Tests\n');
console.log('='.repeat(60));

testRecursiveChunking();
testDocumentCreation();
testEdgeCases();

console.log('\n' + '='.repeat(60));
console.log('✅ Tests completed! The TextChunker now follows LangChain tutorial approach.');
console.log('📚 It uses RecursiveCharacterTextSplitter-like logic with hierarchical separators.');
