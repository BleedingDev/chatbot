import type { ChatPromptTemplate } from "@langchain/core/prompts"
import { pull } from "langchain/hub"
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents"
import { ChatOpenAI } from "@langchain/openai"
import { TAnyToolDefinitionArray, TToolDefinitionMap } from "@/lib/utils/tool-definition"
import { LangChainStream, OpenAIStream, OpenAIStreamCallbacks } from "ai"
import type OpenAI from "openai"
import zodToJsonSchema from "zod-to-json-schema"
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { z } from "zod"
import { RateLimitError } from "openai"
import { DynamicStructuredTool, StructuredToolInterface } from "@langchain/core/tools"

const consumeStream = async (stream: ReadableStream) => {
  const reader = stream.getReader()
  while (true) {
    const { done } = await reader.read()
    if (done) break
  }
}

export function runOpenAICompletion<
  T extends Omit<Parameters<typeof OpenAI.prototype.chat.completions.create>[0], "functions">,
  const TFunctions extends TAnyToolDefinitionArray,
>(
  openai: OpenAI,
  params: T & {
    functions: TFunctions
  },
) {
  let text = ""
  let hasFunction = false

  type TToolMap = TToolDefinitionMap<TFunctions>
  let onTextContent: (text: string, isFinal: boolean) => void = () => {}
  let onError: (error: string) => void = () => {}

  const functionsMap: Record<string, TFunctions[number]> = {}
  for (const fn of params.functions) {
    functionsMap[fn.name] = fn
  }

  let onFunctionCall = {} as any

  const { functions, messages, ...rest } = params

  const aiStreamCallbacks: OpenAIStreamCallbacks = {
    async experimental_onFunctionCall(functionCallPayload) {
      hasFunction = true

      if (!onFunctionCall[functionCallPayload.name]) {
        return
      }

      // we need to convert arguments from z.input to z.output
      // this is necessary if someone uses a .default in their schema
      const zodSchema = functionsMap[functionCallPayload.name].parameters
      const parsedArgs = zodSchema.safeParse(functionCallPayload.arguments)

      if (!parsedArgs.success) {
        throw new Error(`Invalid function call in message. Expected a function call object`)
      }

      onFunctionCall[functionCallPayload.name]?.(parsedArgs.data)
    },
    onToken(token) {
      text += token
      if (text.startsWith("{")) return
      onTextContent(text, false)
    },
    onFinal() {
      if (hasFunction) return
      onTextContent(text, true)
    },
  }

  ;(async () => {
    try {
      const llm = new ChatOpenAI({
        streaming: true,
        temperature: 0,
      })
      const prompt = await pull<ChatPromptTemplate>("hwchase17/openai-functions-agent")
      const tools: StructuredToolInterface[] = functions.map(
        (fn) =>
          new DynamicStructuredTool({
            name: fn.name,
            description: fn.description || "",
            schema: fn.parameters,
            func: (fn as any).func,
          }),
      )

      const agent = await createOpenAIFunctionsAgent({
        llm,
        tools,
        prompt,
      })

      const agentExecutor = new AgentExecutor({
        agent,
        tools,
      })

      const stream = await agentExecutor.stream({
        input: messages.at(-1)?.content,
        chatHistory: messages.slice(0, -1),
      })

      const openAiStream = OpenAIStream(
        (await openai.chat.completions.create({
          ...rest,
          messages,
          stream: true,
          functions: functions.map((fn) => ({
            name: fn.name,
            description: fn.description,
            parameters: zodToJsonSchema(fn.parameters) as Record<string, unknown>,
          })),
        })) as any,
        aiStreamCallbacks,
      )
      consumeStream(stream)
    } catch (e) {
      console.error(e)
      if (e instanceof Error && e.name === "InsufficientQuotaError") {
        onError("The app has not enough credits, please try again later.")
      } else {
        onError("Unknown error occurred. We're working on it.")
      }
    }
  })()

  return {
    onTextContent: (callback: (text: string, isFinal: boolean) => void | Promise<void>) => {
      onTextContent = callback
    },
    onError: (callback: (error: string) => void | Promise<void>) => {
      onError = callback
    },
    onFunctionCall: <TName extends TFunctions[number]["name"]>(
      name: TName,
      callback: (
        args: z.output<
          TName extends keyof TToolMap
            ? TToolMap[TName] extends infer TToolDef
              ? TToolDef extends TAnyToolDefinitionArray[number]
                ? TToolDef["parameters"]
                : never
              : never
            : never
        >,
      ) => void | Promise<void>,
    ) => {
      onFunctionCall[name] = callback
    },
  }
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const formatNumber = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value)

export const runAsyncFnWithoutBlocking = (fn: (...args: any) => Promise<any>) => {
  fn()
}

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// Fake data
export function getStockPrice(name: string) {
  let total = 0
  for (let i = 0; i < name.length; i++) {
    total = (total + name.charCodeAt(i) * 9999121) % 9999
  }
  return total / 100
}
