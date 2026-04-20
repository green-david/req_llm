defmodule ReqLLM.Providers.Anthropic.ResponseBuilder do
  @moduledoc """
  Anthropic-specific ResponseBuilder implementation.

  Handles Anthropic's specific requirements:
  - Content blocks must be non-empty when tool_calls are present
  - Maps `tool_use` finish reason to `:tool_calls`

  This fixes bug #269 where streaming tool-call-only responses
  produced empty content blocks that Anthropic's API rejected.
  """

  @behaviour ReqLLM.Provider.ResponseBuilder

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Provider.Defaults.ResponseBuilder, as: DefaultBuilder
  alias ReqLLM.StreamChunk

  @impl true
  def build_response(chunks, metadata, opts) do
    chunks = backfill_thinking_signatures(chunks)
    meta_chunk = reasoning_details_meta_chunk(chunks)

    with {:ok, response} <- DefaultBuilder.build_response(chunks ++ [meta_chunk], metadata, opts) do
      {:ok, ensure_non_empty_content(response)}
    end
  end

  defp ensure_non_empty_content(%{message: %{tool_calls: tc, content: []}} = response)
       when is_list(tc) and tc != [] do
    content = [ContentPart.text("")]
    put_in(response.message.content, content)
  end

  defp ensure_non_empty_content(response), do: response

  defp backfill_thinking_signatures(chunks) do
    thinking_signatures_by_content_block_index =
      chunks
      |> Enum.flat_map(fn
        %StreamChunk{
          type: :meta,
          metadata: %{thinking_signature: signature, content_block_index: index}
        } ->
          [{index, signature}]

        _ ->
          []
      end)
      |> Map.new()

    Enum.map(chunks, fn
      %StreamChunk{type: :thinking, metadata: %{content_block_index: index}} = chunk ->
        updated_metadata =
          case Map.get(thinking_signatures_by_content_block_index, index) do
            nil -> %{chunk.metadata | signature: nil, encrypted?: false}
            signature -> %{chunk.metadata | signature: signature, encrypted?: true}
          end

        %{chunk | metadata: updated_metadata}

      chunk ->
        chunk
    end)
  end

  defp reasoning_details_meta_chunk(chunks) do
    reasoning_details =
      chunks
      |> Enum.filter(fn %StreamChunk{type: type} -> type == :thinking end)
      |> Enum.chunk_by(fn %StreamChunk{metadata: metadata} -> metadata.content_block_index end)
      |> Enum.with_index()
      |> Enum.map(fn {thinking_chunks, idx} ->
        # All chunks in the same content block should share the same signature (in theory), so we
        # just sample the first one to use.
        [%StreamChunk{metadata: %{signature: signature}} | _] = thinking_chunks
        text = Enum.map_join(thinking_chunks, "", & &1.text)

        %ReqLLM.Message.ReasoningDetails{
          text: text,
          signature: signature,
          encrypted?: signature != nil,
          provider: :anthropic,
          format: "anthropic-thinking-v1",
          index: idx,
          provider_data: %{"type" => "thinking"}
        }
      end)

    StreamChunk.meta(%{reasoning_details: reasoning_details})
  end
end
