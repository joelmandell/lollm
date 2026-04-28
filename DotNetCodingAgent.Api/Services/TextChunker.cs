namespace DotNetCodingAgent.Api.Services;

public static class TextChunker
{
    public static IReadOnlyList<string> Chunk(string text, int chunkSize = 1400, int overlap = 200)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        var normalized = text.Trim();
        if (normalized.Length <= chunkSize)
        {
            return [normalized];
        }

        var chunks = new List<string>();
        var start = 0;

        while (start < normalized.Length)
        {
            var desiredEnd = Math.Min(start + chunkSize, normalized.Length);
            var end = desiredEnd;

            if (desiredEnd < normalized.Length)
            {
                var lastWhitespace = normalized.LastIndexOf(' ', desiredEnd - 1, desiredEnd - start);
                if (lastWhitespace > start + chunkSize / 2)
                {
                    end = lastWhitespace;
                }
            }

            var chunk = normalized[start..end].Trim();
            if (!string.IsNullOrWhiteSpace(chunk))
            {
                chunks.Add(chunk);
            }

            if (end >= normalized.Length)
            {
                break;
            }

            start = Math.Max(0, end - overlap);
        }

        return chunks;
    }
}
