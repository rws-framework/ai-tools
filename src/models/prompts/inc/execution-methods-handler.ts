import { 
    IRWSPromptRequestExecutor,
    IRWSSinglePromptRequestExecutor,
    IRWSPromptStreamExecutor,
    ILLMChunk,
    IAITool
} from './types';

export interface IPromptInstance {
    readOutput(): string;
    readInput(): any[];
    getInput(): any[];
    setSentInput(input: any[]): void;
    injestOutput(content: string): void;
}

export class ExecutionMethodsHandler {
    async requestWith(
        promptInstance: IPromptInstance,
        executor: IRWSPromptRequestExecutor, 
        intruderPrompt: string = null, 
        debugVars: any = {}, 
        tools?: IAITool[]
    ): Promise<void> {
        promptInstance.setSentInput(promptInstance.getInput());
        const returnedRWS = await executor.promptRequest(promptInstance as any, { intruderPrompt, debugVars, tools });
        promptInstance.injestOutput(returnedRWS.readOutput());
    }

    async singleRequestWith(
        promptInstance: IPromptInstance,
        executor: IRWSSinglePromptRequestExecutor, 
        intruderPrompt: string = null, 
        ensureJson: boolean = false, 
        tools?: IAITool[]
    ): Promise<void> {
        await executor.singlePromptRequest(promptInstance as any, { intruderPrompt, ensureJson, tools });
        promptInstance.setSentInput(promptInstance.getInput());
    }

    async streamWith(
        promptInstance: IPromptInstance,
        executor: IRWSPromptStreamExecutor, 
        read: (chunk: ILLMChunk) => void, 
        end: () => void = () => { }, 
        debugVars: any = {}, 
        tools?: IAITool[]
    ): Promise<any> {
        promptInstance.setSentInput(promptInstance.getInput());
        return executor.promptStream(promptInstance as any, read, end, { debugVars, tools });
    }
}
