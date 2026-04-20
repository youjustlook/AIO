import os, json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
load_dotenv()

# df = pd.read_csv('combined_sampled_queries.csv')
# print(df.head())

client = genai.Client(api_key='FILL IN')


grounding_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(
    tools=[grounding_tool],
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)
MODEL_NAME = "gemini-2.5-flash"
MODEL_INPUT_PRICE = 0.3 # per 1M tokens
MODEL_OUTPUT_PRICE = 2.5 # per 1M tokens

#def add_citations(response):
#    text = response.text
#    supports = response.candidates[0].grounding_metadata.grounding_supports
#    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
 #   sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

#    for support in sorted_supports:
#        end_index = support.segment.end_index
#        if support.grounding_chunk_indices:
#            # Create citation string like [1](link1)[2](link2)
#            citation_links = []
#            for i in support.grounding_chunk_indices:
#                if i < len(chunks):
#                    uri = chunks[i].web.uri
#                    citation_links.append(f"[{i + 1}]({uri})")

#            citation_string = ", ".join(citation_links)
#            text = text[:end_index] + citation_string + text[end_index:]

#    return text

def add_citations(response):
    text = response.text

    try:
        candidate = response.candidates[0]
        grounding = candidate.grounding_metadata

        # If no grounding data exists, just return original text
        if not grounding or not grounding.grounding_supports or not grounding.grounding_chunks:
            return text

        supports = grounding.grounding_supports
        chunks = grounding.grounding_chunks

        # Sort supports by end_index in descending order
        sorted_supports = sorted(
            supports,
            key=lambda s: getattr(getattr(s, "segment", None), "end_index", None) or 0,
            reverse=True
        )

        for support in sorted_supports:
            if not support.grounding_chunk_indices:
                continue

            end_index = support.segment.end_index

            citation_links = []
            for idx in support.grounding_chunk_indices:
                if idx < len(chunks) and getattr(chunks[idx], "web", None):
                    uri = chunks[idx].web.uri
                    citation_links.append(f"[{idx + 1}]({uri})")

            if citation_links:
                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

    except Exception as e:
        print(f"Warning: citation handling failed: {e}")
        return text

    return text


'''def get_completion(_id, query):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=query,
        config=config,
    )
    response_dict = response.model_dump()
    # print(response_dict.keys())
    # print(response_dict['sdk_http_response'].keys())

    search_tool_used = False
    if response.usage_metadata.tool_use_prompt_token_count:
        search_tool_used = True

    c = {
        'id': _id, 'query': query,
        'text': response.text,
        'text_with_citations': (add_citations(response) if search_tool_used else None),
        'web_search_queries': (response_dict['candidates'][0]['grounding_metadata']['web_search_queries'] if search_tool_used else None),
        'web_sources': (response_dict['candidates'][0]['grounding_metadata']['grounding_chunks'] if search_tool_used else None),
        'web_supported_text': (response_dict['candidates'][0]['grounding_metadata']['grounding_supports'] if search_tool_used else None),
        'input_tokens': response.usage_metadata.prompt_token_count,
        'search_tokens': response.usage_metadata.tool_use_prompt_token_count or 0,
        'output_tokens': response.usage_metadata.candidates_token_count
    }
    assert response.usage_metadata.total_token_count == c['input_tokens'] + c['search_tokens'] + c['output_tokens']
    c['total_cost'] = (c['input_tokens'] + c['search_tokens']) / 1_000_000 * MODEL_INPUT_PRICE + c['output_tokens'] / 1_000_000 * MODEL_OUTPUT_PRICE

    return c'''

def get_completion(_id, query):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=query,
        config=config,
    )

    response_dict = response.model_dump()

    # Was a search tool used?
    search_tool_used = bool(response.usage_metadata.tool_use_prompt_token_count)

    # Safely extract candidate + grounding
    candidate = response_dict.get('candidates', [{}])[0]
    grounding = candidate.get('grounding_metadata') or {}

    web_search_queries = None
    web_sources = None
    web_supported_text = None
    text_with_citations = None

    if search_tool_used and grounding:
        web_search_queries = grounding.get('web_search_queries')
        web_sources = grounding.get('grounding_chunks')
        web_supported_text = grounding.get('grounding_supports')

        # Only try to add citations if supports actually exist
        if web_supported_text and web_sources:
            text_with_citations = add_citations(response)

    c = {
        'id': _id,
        'query': query,
        'text': response.text,
        'text_with_citations': text_with_citations,
        'web_search_queries': web_search_queries,
        'web_sources': web_sources,
        'web_supported_text': web_supported_text,
        'input_tokens': response.usage_metadata.prompt_token_count,
        'search_tokens': response.usage_metadata.tool_use_prompt_token_count or 0,
        'output_tokens': response.usage_metadata.candidates_token_count
    }

    # Safety: avoid crashing your loop on a token mismatch

    try:
        assert response.usage_metadata.total_token_count == (
            c['input_tokens'] + c['search_tokens'] + c['output_tokens']
        )
    except Exception:
        print(f"Warning: token mismatch on ID {_id}")
        
    if c['output_tokens'] is not None:
        c['total_cost'] = (
            (c['input_tokens'] + c['search_tokens']) / 1_000_000 * MODEL_INPUT_PRICE
            + c['output_tokens'] / 1_000_000 * MODEL_OUTPUT_PRICE
        )
       

    return c




if __name__ == '__main__':
    queries = [
        '1+1=', # no web search
        "who won the 7 man elimination chamber match" # uses web search
    ]
    query_filename = '.../AIO_Benchmark_Dataset.csv'
    output_filename = 'SIGIR_Gemini_data.jsonl'

    df = pd.read_csv(query_filename)
    try:
        with open(output_filename, "r") as f:
            completions = [json.loads(line) for line in f]
    except FileNotFoundError:
        open(output_filename, "w").close()
        completions = []
    
    total_cost, uses_search = 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Getting completions"):
        if row['ID'] not in [c['id'] for c in completions]:
            # print(row)
            c = get_completion(row['ID'], row['ORCAS Label'])
            with open(output_filename, "a") as f:
                f.write(json.dumps(c) + "\n")
        
            completions.append(c)
        else:
            pass
            # print(f'id={row['ID']} skipped')
            # total_cost += c['total_cost']
            # uses_search += (1 if c['search_tokens'] else 0)
    
    # with open("sample_query_completions.json", "w") as json_file:
    #     json.dump(completions, json_file, indent=4)
    
    # print('% that uses search:', uses_search / len(df))
    # print('total cost:', total_cost)
    # print('avg cost per query:', total_cost / len(df))

    # TO-DO:
    # 1. Add more fields: 
