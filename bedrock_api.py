import json
import boto3

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


def call_claude_3_sonnet(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


# Call Titan model
def call_titan(prompt):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-text-lite-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body)

    results = response_body.get("results")[0].get("outputText")
    return results


def claude_2_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"


def call_claude_2(prompt):
    prompt_config = {
        "prompt": claude_2_prompt_format(prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Bedrock api call to stable diffusion
def generate_image(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


def lambda_handler(event, context):

    data_string = event["body"]
    input_data = json.loads(data_string)
    model = input_data["model"]
    prompt = input_data["prompt"]

    # check for stable diffusion style
    if "style" in input_data:
        style = input_data["style"]
    else:
        style = "None"

    # You will need to write prompts to fill out the code to call the correct model based on input

    if model == "titan":
        generated_text = call_titan(prompt)
        response_body = json.dumps({"text": generated_text})
    elif model == "claude":
        generated_text = call_claude_3_sonnet(prompt)
        response_body = json.dumps({"text": generated_text})
    elif model == "stable_diffusion":
        # Call stable diffusion model
        generated_image = generate_image(prompt, style)
        response_body = json.dumps({"image": generated_image})
    else:
        body = f"Error {model} is not a valid model, try another one."
        response_body = json.dumps({"error": body})

    # DO NOT CHANGE CODE BELOW
    return {"statusCode": 200, "body": response_body}
