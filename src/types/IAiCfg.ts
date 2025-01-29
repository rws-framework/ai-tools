import { IAppConfig } from "@rws-framework/server";

export interface IAiCfg extends IAppConfig {
    aws_bedrock_region?: string;
    aws_access_key?: string;
    aws_secret_key?: string;
    cohere_api_key?: string;
}