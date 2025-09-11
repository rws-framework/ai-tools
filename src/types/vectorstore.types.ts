/**
 * Vector store configuration interfaces
 */
export interface IVectorStoreConfig {
    type: 'memory';
    maxResults?: number;
    autoSave?: boolean;
    similarityThreshold?: number;
}

export interface IDocumentChunk {
    id: string;
    content: string;
    embedding?: number[];
    metadata?: {
        documentId: string;
        chunkIndex: number;
        source?: string;
        title?: string;
        knowledgeId?: string;
        [key: string]: any;
    };
}
