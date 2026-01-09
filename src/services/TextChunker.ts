/**
 * Text chunking utility following LangChain tutorial approach
 * Uses RecursiveCharacterTextSplitter-like logic with token optimization
 */
export class TextChunker {
    /**
     * Conservative token estimation based on character count
     * Uses a slightly more accurate ratio than the original 4 chars per token
     */
    static estimateTokens(text: string): number {
        // More accurate estimation: ~3.7 characters per token on average
        return Math.ceil(text.length / 3.7);
    }

    /**
     * Default separators following LangChain RecursiveCharacterTextSplitter approach
     * Ordered by preference for breaking text
     */
    private static readonly DEFAULT_SEPARATORS = [
        '\n\n',  // Double newlines (paragraphs)
        '\n',    // Single newlines
        '. ',    // Sentence endings
        '! ',    // Exclamation sentence endings
        '? ',    // Question sentence endings
        '; ',    // Semicolons
        ', ',    // Commas
        ' ',     // Spaces (words)
        ''       // Character level (fallback)
    ];

    /**
     * Chunk text using RecursiveCharacterTextSplitter approach from LangChain tutorial
     * Recursively tries separators until chunks fit within token limits
     * 
     * @param text - The text to chunk
     * @param maxTokens - Maximum tokens per chunk (default: 450 to be safe under 512)
     * @param overlap - Number of characters to overlap between chunks
     * @param separators - Custom separators (uses defaults if not provided)
     * @returns Array of text chunks
     */
    static chunkText(
        text: string, 
        maxTokens: number = 450, 
        overlap: number = 50,
        separators: string[] = TextChunker.DEFAULT_SEPARATORS
    ): string[] {
        if (!text || text.trim().length === 0) {
            return [];
        }

        // If text is already within limits, return as-is
        const estimatedTokens = this.estimateTokens(text);
        if (estimatedTokens <= maxTokens) {
            return [text.trim()];
        }

        console.log(`[TextChunker] Chunking text: ${text.length} chars, estimated ${estimatedTokens} tokens, max ${maxTokens} tokens per chunk`);

        // Convert token limit to character limit (conservative)
        const maxCharsPerChunk = Math.floor(maxTokens * 3.5);

        // Use recursive splitting approach like LangChain tutorial
        return this.recursiveSplit(text, maxCharsPerChunk, overlap, separators);
    }

    /**
     * Recursive splitting approach following LangChain's RecursiveCharacterTextSplitter
     * Tries each separator in order until chunks fit within limits
     */
    private static recursiveSplit(
        text: string, 
        maxChars: number, 
        overlap: number, 
        separators: string[]
    ): string[] {
        const finalChunks: string[] = [];
        
        // Get the first separator to try
        const separator = separators[0];
        let newSeparators: string[];
        
        if (separator === '') {
            // Character-level splitting (last resort)
            newSeparators = [];
        } else {
            // Continue with remaining separators for recursive calls
            newSeparators = separators.slice(1);
        }

        // Split text by current separator
        const splits = this.splitTextBySeparator(text, separator);

        // Process each split
        for (const split of splits) {
            if (split.length <= maxChars) {
                // Split fits within limit, add it
                finalChunks.push(split);
            } else {
                // Split is too large, need to recursively split further
                if (newSeparators.length === 0) {
                    // No more separators, force split by character limit
                    finalChunks.push(...this.forceSplitByCharacterLimit(split, maxChars, overlap));
                } else {
                    // Recursively split with next separator
                    finalChunks.push(...this.recursiveSplit(split, maxChars, overlap, newSeparators));
                }
            }
        }

        // Merge small chunks and add overlaps (like tutorial's approach)
        return this.mergeChunksWithOverlap(finalChunks, maxChars, overlap);
    }

    /**
     * Split text by a specific separator (preserving separator where appropriate)
     */
    private static splitTextBySeparator(text: string, separator: string): string[] {
        if (separator === '') {
            // Character-level split
            return text.split('');
        }

        if (!text.includes(separator)) {
            return [text];
        }

        // Split by separator and preserve meaningful separators
        const splits = text.split(separator);
        const result: string[] = [];

        for (let i = 0; i < splits.length; i++) {
            const split = splits[i];
            
            if (i === splits.length - 1) {
                // Last split, don't add separator
                if (split.trim()) {
                    result.push(split.trim());
                }
            } else {
                // Add separator back for meaningful breaks
                const withSeparator = split + (this.shouldPreserveSeparator(separator) ? separator : '');
                if (withSeparator.trim()) {
                    result.push(withSeparator.trim());
                }
            }
        }

        return result.filter(s => s.length > 0);
    }

    /**
     * Determine if separator should be preserved in the text
     */
    private static shouldPreserveSeparator(separator: string): boolean {
        // Preserve sentence endings and meaningful punctuation
        return ['. ', '! ', '? ', '; '].includes(separator);
    }

    /**
     * Force split by character limit when no separators work
     */
    private static forceSplitByCharacterLimit(text: string, maxChars: number, overlap: number): string[] {
        const chunks: string[] = [];
        let position = 0;

        while (position < text.length) {
            let endPosition = position + maxChars;
            
            if (endPosition >= text.length) {
                // Last chunk
                const lastChunk = text.substring(position).trim();
                if (lastChunk) {
                    chunks.push(lastChunk);
                }
                break;
            }

            chunks.push(text.substring(position, endPosition).trim());
            position = endPosition - overlap;

            // Ensure we don't go backwards
            if (position < 0) position = 0;
        }

        return chunks.filter(chunk => chunk.length > 0);
    }

    /**
     * Merge small chunks and add overlaps following tutorial approach
     */
    private static mergeChunksWithOverlap(chunks: string[], maxChars: number, overlap: number): string[] {
        if (chunks.length === 0) return [];
        
        const mergedChunks: string[] = [];
        let currentChunk = '';

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            
            // Use array for efficient string building
            const parts = currentChunk ? [currentChunk, chunk] : [chunk];
            const combined = parts.join(' ');
            
            if (combined.length <= maxChars) {
                // Can merge
                currentChunk = combined;
            } else {
                // Can't merge, save current chunk and start new one
                if (currentChunk) {
                    mergedChunks.push(currentChunk.trim());
                }
                currentChunk = chunk;
            }
        }

        // Add final chunk
        if (currentChunk.trim()) {
            mergedChunks.push(currentChunk.trim());
        }

        // Add overlaps between chunks
        return this.addOverlapsBetweenChunks(mergedChunks, overlap);
    }

    /**
     * Add overlaps between chunks like the tutorial approach
     */
    private static addOverlapsBetweenChunks(chunks: string[], overlap: number): string[] {
        if (chunks.length <= 1 || overlap <= 0) {
            return chunks;
        }

        const chunksWithOverlap: string[] = [];
        
        for (let i = 0; i < chunks.length; i++) {
            let chunkWithOverlap = chunks[i];
            
            // Add overlap from previous chunk at the beginning
            if (i > 0) {
                const prevOverlap = this.extractOverlap(chunks[i - 1], overlap);
                if (prevOverlap && !chunkWithOverlap.startsWith(prevOverlap)) {
                    chunkWithOverlap = prevOverlap + ' ' + chunkWithOverlap;
                }
            }
            
            chunksWithOverlap.push(chunkWithOverlap.trim());
        }

        return chunksWithOverlap;
    }

    /**
     * Extract overlap text from the end of a chunk
     */
    private static extractOverlap(text: string, overlapLength: number): string {
        if (text.length <= overlapLength) {
            return text;
        }

        // Try to break at word boundary for overlap
        let startPosition = text.length - overlapLength;
        while (startPosition < text.length && text[startPosition] !== ' ') {
            startPosition++;
        }

        if (startPosition >= text.length) {
            // No word boundary found, just take last characters
            return text.substring(text.length - overlapLength);
        }

        return text.substring(startPosition + 1); // +1 to skip the space
    }

    /**
     * Truncate text to fit within token limit (utility method)
     */
    static truncateText(text: string, maxTokens: number): string {
        const maxChars = Math.floor(maxTokens * 3.5);
        
        if (text.length <= maxChars) {
            return text;
        }

        // Try to truncate at word boundary
        let truncatePosition = maxChars;
        while (truncatePosition > maxChars * 0.8 && text[truncatePosition] !== ' ') {
            truncatePosition--;
        }

        if (truncatePosition <= maxChars * 0.8) {
            // No word boundary found, truncate at character limit
            truncatePosition = maxChars;
        }

        return text.substring(0, truncatePosition).trim();
    }

    /**
     * Create documents from text chunks (tutorial-style helper)
     * Similar to how the tutorial creates Document objects from splits
     */
    static createDocumentsFromChunks(
        text: string, 
        metadata: Record<string, any> = {},
        maxTokens: number = 450,
        overlap: number = 50
    ): Array<{ pageContent: string; metadata: Record<string, any> }> {
        const chunks = this.chunkText(text, maxTokens, overlap);
        
        return chunks.map((chunk, index) => ({
            pageContent: chunk,
            metadata: {
                ...metadata,
                chunkIndex: index,
                id: `${metadata.documentId || 'doc'}_chunk_${index}`,
                totalChunks: chunks.length
            }
        }));
    }
}
