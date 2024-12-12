import { EmbeddingsInterface } from '@rws-framework/ai-tools/node_modules/@langchain/core/embeddings';
import { Injectable } from '@rws-framework/ai-tools/node_modules/@rws-framework/server/nest';  

import RWSVectorStore, { VectorDocType } from '@rws-framework/ai-tools/src/models/convo/VectorStore';

@Injectable()
class VectorStoreService
{
    async createStore(docs: VectorDocType, embeddings: EmbeddingsInterface): Promise<RWSVectorStore>
    {        
        return await (new RWSVectorStore(docs, embeddings)).init();
    }    
}

export { VectorStoreService };