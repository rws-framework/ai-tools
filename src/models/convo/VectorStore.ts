import { FaissStore } from '@rws-framework/ai-tools/node_modules/@langchain/community/vectorstores/faiss';
import { EmbeddingsInterface } from '@rws-framework/ai-tools/node_modules/@langchain/core/embeddings';
import { Document } from '@rws-framework/ai-tools/node_modules/@langchain/core/documents';

type VectorDocType = Document<Record<string, any>>[];

export default class RWSVectorStore 
{
    private faiss: FaissStore;
    private docs: VectorDocType;
    private embeddings: EmbeddingsInterface;

    constructor(docs: VectorDocType, embeddings: EmbeddingsInterface){
        this.docs = docs;
        this.embeddings = embeddings;
    }

    async init(): Promise<RWSVectorStore>
    {
        this.faiss = await FaissStore.fromDocuments(this.docs, this.embeddings);

        return this;
    }

    getFaiss(): FaissStore
    {
        return this.faiss;
    }
}

export {
    VectorDocType
};