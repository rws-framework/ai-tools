import 'reflect-metadata';

import { ConsoleService, RWSConfigService, RWSErrorCodes} from '@rws-framework/server';
import { InjectServices } from '@rws-framework/server/src/services/_inject';
import RWSPrompt from '../prompts/_prompt';
import { IRWSPromptJSON, ILLMChunk } from '../../types/IPrompt';
import {VectorStoreService} from '../../services/VectorStoreService';
import RWSVectorStore, { VectorDocType } from './VectorStore';

import { Document } from '@langchain/core/documents';
import { UnstructuredLoader } from '@langchain/community/document_loaders/fs/unstructured';

import { BaseChatModel  } from "@langchain/core/language_models/chat_models";
import { BaseLanguageModelInterface, BaseLanguageModelInput } from '@langchain/core/language_models/base';
import { Runnable } from '@langchain/core/runnables';
import { BaseMessage } from '@langchain/core/messages';


import { v4 as uuid } from 'uuid';
import xml2js from 'xml2js';
import fs from 'fs';
import path from 'path';
import { IAiCfg } from '../../types/IAiCfg';


interface ISplitterParams {
    chunkSize: number
    chunkOverlap: number
    separators: string[]
}

const logConvo = (txt: string) => {
    ConsoleService.rwsLog(ConsoleService.color().blueBright(txt));
};

interface IBaseLangchainHyperParams {
    temperature: number;
    topK: number;
    topP: number;
    maxTokens: number;
}

interface IConvoDebugXMLData {
    conversation: {
        $: {
            id: string
            [key: string]: string
        },
        message: IRWSPromptJSON[]        
    }
}

interface IConvoDebugXMLOutput {
    xml: IConvoDebugXMLData,
    path: string
}

interface IChainCallOutput {
    text: string
}

interface IEmbeddingsHandler<T extends object> {
    generateEmbeddings: (text?: string) => Promise<T>
    storeEmbeddings: (embeddings: any, convoId: string) => Promise<void>
}

type LLMType = BaseLanguageModelInterface | Runnable<BaseLanguageModelInput, string> | Runnable<BaseLanguageModelInput, BaseMessage>;

@InjectServices([VectorStoreService])
class EmbedLoader<LLMChat extends BaseChatModel> {
    private loader: UnstructuredLoader;
    private embeddings: IEmbeddingsHandler<any>;

    private docs: Document[] = [];
    private _initiated = false;
    private convo_id: string;        
    private llmChat: LLMChat;
    private chatConstructor: new (config: any) => LLMChat;
    private thePrompt: RWSPrompt;

    vectorStoreService: VectorStoreService;
    configService: RWSConfigService<IAiCfg>;

    public _baseSplitterParams: ISplitterParams;    
    
    constructor(
        chatConstructor: new (config: any) => LLMChat, 
        embeddings: IEmbeddingsHandler<any> | null = null, 
        convoId: string | null = null, 
        baseSplitterParams: ISplitterParams = {
            chunkSize: 400, 
            chunkOverlap: 80, 
            separators: ['/n/n','.']
        }
    ) {
        this.embeddings = embeddings;
        if(convoId === null) {
            this.convo_id = EmbedLoader.uuid();
        } else {
            this.convo_id = convoId;
        }                        
        this.chatConstructor = chatConstructor;    
        this._baseSplitterParams = baseSplitterParams;  
    }

    static uuid(): string
    {
        return uuid();
    }    

    getId(): string {
        return this.convo_id;
    }

    getDocs(): VectorDocType
    {
        return this.docs;
    }

    isInitiated(): boolean 
    {
        return this._initiated;
    }

    getChat(): LLMChat
    {
        return this.llmChat;
    }

    private avgDocLength = (documents: Document[]): number => {
        return documents.reduce((sum, doc: Document) => sum + doc.pageContent.length, 0) / documents.length;
    };

    async splitDocs(filePath: string, params: ISplitterParams): Promise<RWSVectorStore>
    {

        if(!this.embeddings){
            throw new Error('No embeddings provided for ConvoLoader\'s constructor. ConvoLoader.splitDocs aborting...');
        }

        const splitDir = EmbedLoader.debugSplitDir(this.getId());
        const finalDocs = [];

        if(!fs.existsSync(splitDir)){
            console.log(`Split dir ${ConsoleService.color().magentaBright(splitDir)} doesn't exist. Splitting docs...`);
            this.loader = new UnstructuredLoader(filePath);

            // this.docSplitter = new RecursiveCharacterTextSplitter({
            //     chunkSize: params.chunkSize, // The size of the chunk that should be split.
            //     chunkOverlap: params.chunkOverlap, // Adding overalap so that if a text is broken inbetween, next document may have part of the previous document 
            //     separators: params.separators // In this case we are assuming that /n/n would mean one whole sentence. In case there is no nearing /n/n then "." will be used instead. This can be anything that helps derive a complete sentence .
            // });

            fs.mkdirSync(splitDir, { recursive: true });
            
            const orgDocs = await this.loader.load();
            const splitDocs: any[] = [];//await this.docSplitter.splitDocuments(orgDocs);

            const avgCharCountPre = this.avgDocLength(orgDocs);
            const avgCharCountPost = this.avgDocLength(splitDocs);

            logConvo(`Average length among ${orgDocs.length} documents loaded is ${avgCharCountPre} characters.`);
            logConvo(`After the split we have ${splitDocs.length} documents more than the original ${orgDocs.length}.`);
            logConvo(`Average length among ${orgDocs.length} documents (after split) is ${avgCharCountPost} characters.`);

            let i = 0;
            splitDocs.forEach((doc: Document) => {
                finalDocs.push(doc);
                fs.writeFileSync(this.debugSplitFile(i), doc.pageContent);
                i++;
            });
        }else{
            const splitFiles = fs.readdirSync(splitDir);
            
            for(const filePath of splitFiles) {
                const txt = fs.readFileSync(splitDir + '/' + filePath, 'utf-8');
                finalDocs.push(new Document({ pageContent: txt }));              
            }
        }
        
        return await this.vectorStoreService.createStore(finalDocs, await this.embeddings.generateEmbeddings());
    }

    async similaritySearch(query: string, splitCount: number, store: RWSVectorStore): Promise<string>
    {
        console.log('Store is ready. Searching for embedds...');            
        const texts = await store.getFaiss().similaritySearchWithScore(`${query}`, splitCount);
        console.log('Found best parts: ' + texts.length);
        return texts.map(([doc, score]: [any, number]) => `${doc['pageContent']}`).join('\n\n');    
    }
    
    private async debugCall(debugCallback: (debugData: IConvoDebugXMLData) => Promise<IConvoDebugXMLData> = null)
    {
        try {
            const debug = this.initDebugFile();

            let callData: IConvoDebugXMLData = debug.xml;

            callData.conversation.message.push(this.thePrompt.toJSON());

            if(debugCallback){
                callData = await debugCallback(callData);
            }

            this.debugSave(callData);
        
        }catch(error: Error | unknown){
            console.log(error);
        }
    }


    async waitForInit(): Promise<EmbedLoader<LLMChat> | null>
    {
        const _self = this;
        return new Promise((resolve, reject)=>{
            let i = 0;

            const interval: NodeJS.Timeout = setInterval(() => {
                if(this.isInitiated()){
                    clearInterval(interval);
                    resolve(_self);
                }

                if(i>9){
                    clearInterval(interval);
                    reject(null);
                }

                i++;
            }, 300);            
        });
    }  

    async setPrompt(prompt: RWSPrompt){
        this.thePrompt = prompt;
    }

    private parseXML(xml: string, callback: (err: Error, result: any) => void): xml2js.Parser
    {
        const parser = new xml2js.Parser();        

        parser.parseString(xml, callback);
        return parser;
    }

    static debugConvoDir(id: string){
        return path.resolve(process.cwd(), 'debug', 'conversations', id);
    }

    static debugSplitDir(id: string){
        return path.resolve(process.cwd(), 'debug', 'conversations', id, 'split');
    }
    
    public debugConvoFile(){
        return `${EmbedLoader.debugConvoDir(this.getId())}/conversation.xml`;
    }    

    public debugSplitFile(i: number){
        return `${EmbedLoader.debugSplitDir(this.getId())}/${i}.splitfile`;
    }    

    private initDebugFile(): IConvoDebugXMLOutput
    {
        let xmlContent: string;
        let debugXML: IConvoDebugXMLData = null;

        const convoDir = EmbedLoader.debugConvoDir(this.getId());

        if(!fs.existsSync(convoDir)){
            fs.mkdirSync(convoDir, { recursive: true });
        }

        const convoFilePath = this.debugConvoFile();

        if(!fs.existsSync(convoFilePath)){
            xmlContent = '<conversation id="conversation"></conversation>';

            fs.writeFileSync(convoFilePath, xmlContent);
        }else{
            xmlContent = fs.readFileSync(convoFilePath, 'utf-8');
        }

        this.parseXML(xmlContent, (error: Error, result) => {            
            debugXML = result;
        });

        if(!debugXML.conversation.message){
            debugXML.conversation.message = [];
        }

        return { xml: debugXML, path: convoFilePath };
    }

    private debugSave(xml: IConvoDebugXMLData): void
    {        
        const builder = new xml2js.Builder();
        fs.writeFileSync(this.debugConvoFile(), builder.buildObject(xml), 'utf-8');
    }

}

export { EmbedLoader, IChainCallOutput, IConvoDebugXMLData, IEmbeddingsHandler, ISplitterParams, IBaseLangchainHyperParams };
