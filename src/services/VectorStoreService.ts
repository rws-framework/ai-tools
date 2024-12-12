import { EmbeddingsInterface } from '@langchain/core/embeddings';
import { Injectable } from '@rws-framework/server/nest';  

import RWSVectorStore, { VectorDocType } from '../models/convo/VectorStore';

@Injectable()
class VectorStoreService
{
    async createStore(docs: VectorDocType, embeddings: EmbeddingsInterface): Promise<RWSVectorStore>
    {        
        return await (new RWSVectorStore(docs, embeddings)).init();
    }    
}

export { VectorStoreService };