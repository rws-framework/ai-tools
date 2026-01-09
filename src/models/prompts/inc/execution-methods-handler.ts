import { 
    IRWSPromptRequestExecutor,
    IRWSSinglePromptRequestExecutor,
    IRWSPromptStreamExecutor,
    ILLMChunk,
    IAITool
} from './types';

export interface IPromptInstance {
    readOutput(): string | object;
    readInput(): any[];
    getInput(): any[];
    setSentInput(input: any[]): void;
    injestOutput(content: string | object): void;
}

export class ExecutionMethodsHandler {
    async requestWith(
        promptInstance: IPromptInstance,
        executor: IRWSPromptRequestExecutor, 
        intruderPrompt: string = null, 
        debugVars: any = {}, 
        tools?: IAITool[]
    ): Promise<void> {
        // Create snapshot of current input to prevent race conditions
        const inputSnapshot = [...promptInstance.getInput()];
        promptInstance.setSentInput(inputSnapshot);
        
        const returnedRWS = await executor.promptRequest(promptInstance as any, { intruderPrompt, debugVars, tools });
        
        // Safely ingest output
        const output = returnedRWS.readOutput();
        if (output !== null && output !== undefined) {
            promptInstance.injestOutput(output);
        }
    }

    async singleRequestWith(
        promptInstance: IPromptInstance,
        executor: IRWSSinglePromptRequestExecutor, 
        intruderPrompt: string = null, 
        ensureJson: boolean = false, 
        tools?: IAITool[]
    ): Promise<void> {
        // Create snapshot of current input to prevent race conditions
        const inputSnapshot = [...promptInstance.getInput()];
        
        await executor.singlePromptRequest(promptInstance as any, { intruderPrompt, ensureJson, tools });
        
        // Set the snapshot after execution to maintain consistency
        promptInstance.setSentInput(inputSnapshot);
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
