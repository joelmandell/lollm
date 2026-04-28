using System.Text.RegularExpressions;
using DotNetCodingAgent.Api.Options;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class SqliteKnowledgeRepository(IOptions<KnowledgeOptions> options) : IKnowledgeRepository
{
    private static readonly Regex TokenRegex = new("[a-z0-9\\.\\-\\+#]{3,}", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private readonly string _connectionString = $"Data Source={Path.GetFullPath(options.Value.DatabasePath)}";

    public async Task InitializeAsync(CancellationToken cancellationToken)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            CREATE TABLE IF NOT EXISTS knowledge_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                last_indexed_utc TEXT NULL,
                last_content_hash TEXT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING fts5(
                source_id UNINDEXED,
                source_url UNINDEXED,
                source_title UNINDEXED,
                content
            );
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<KnowledgeSource>> GetSourcesAsync(CancellationToken cancellationToken)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            SELECT id, url, title, last_indexed_utc, last_content_hash, is_active
            FROM knowledge_sources
            ORDER BY url;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;

        var result = new List<KnowledgeSource>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            result.Add(ReadSource(reader));
        }

        return result;
    }

    public async Task<IReadOnlyList<KnowledgeSource>> GetActiveSourcesAsync(CancellationToken cancellationToken)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            SELECT id, url, title, last_indexed_utc, last_content_hash, is_active
            FROM knowledge_sources
            WHERE is_active = 1
            ORDER BY url;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;

        var result = new List<KnowledgeSource>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            result.Add(ReadSource(reader));
        }

        return result;
    }

    public async Task<KnowledgeSource> AddSourceAsync(string url, string? title, CancellationToken cancellationToken)
    {
        var normalizedUrl = NormalizeUrl(url);
        var normalizedTitle = string.IsNullOrWhiteSpace(title) ? normalizedUrl : title.Trim();

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        const string upsertSql = """
            INSERT INTO knowledge_sources (url, title)
            VALUES ($url, $title)
            ON CONFLICT(url) DO UPDATE SET title = excluded.title
            RETURNING id, url, title, last_indexed_utc, last_content_hash, is_active;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = upsertSql;
        command.Parameters.AddWithValue("$url", normalizedUrl);
        command.Parameters.AddWithValue("$title", normalizedTitle);

        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        await reader.ReadAsync(cancellationToken);
        return ReadSource(reader);
    }

    public async Task UpdateSourceChunksAsync(
        long sourceId,
        string title,
        string contentHash,
        DateTimeOffset indexedAtUtc,
        IReadOnlyList<string> chunks,
        CancellationToken cancellationToken)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);
        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(cancellationToken);

        const string urlLookupSql = "SELECT url FROM knowledge_sources WHERE id = $sourceId;";
        await using var urlLookup = connection.CreateCommand();
        urlLookup.Transaction = transaction;
        urlLookup.CommandText = urlLookupSql;
        urlLookup.Parameters.AddWithValue("$sourceId", sourceId);
        var url = (string?)await urlLookup.ExecuteScalarAsync(cancellationToken) ?? string.Empty;

        const string deleteChunksSql = "DELETE FROM knowledge_chunks WHERE source_id = $sourceId;";
        await using var deleteChunks = connection.CreateCommand();
        deleteChunks.Transaction = transaction;
        deleteChunks.CommandText = deleteChunksSql;
        deleteChunks.Parameters.AddWithValue("$sourceId", sourceId);
        await deleteChunks.ExecuteNonQueryAsync(cancellationToken);

        const string insertChunkSql = """
            INSERT INTO knowledge_chunks (source_id, source_url, source_title, content)
            VALUES ($sourceId, $sourceUrl, $sourceTitle, $content);
            """;

        foreach (var chunk in chunks)
        {
            await using var insertChunk = connection.CreateCommand();
            insertChunk.Transaction = transaction;
            insertChunk.CommandText = insertChunkSql;
            insertChunk.Parameters.AddWithValue("$sourceId", sourceId);
            insertChunk.Parameters.AddWithValue("$sourceUrl", url);
            insertChunk.Parameters.AddWithValue("$sourceTitle", title);
            insertChunk.Parameters.AddWithValue("$content", chunk);
            await insertChunk.ExecuteNonQueryAsync(cancellationToken);
        }

        const string updateSourceSql = """
            UPDATE knowledge_sources
            SET title = $title,
                last_indexed_utc = $indexedAtUtc,
                last_content_hash = $contentHash,
                is_active = 1
            WHERE id = $sourceId;
            """;

        await using var updateSource = connection.CreateCommand();
        updateSource.Transaction = transaction;
        updateSource.CommandText = updateSourceSql;
        updateSource.Parameters.AddWithValue("$title", title);
        updateSource.Parameters.AddWithValue("$indexedAtUtc", indexedAtUtc.ToString("O"));
        updateSource.Parameters.AddWithValue("$contentHash", contentHash);
        updateSource.Parameters.AddWithValue("$sourceId", sourceId);
        await updateSource.ExecuteNonQueryAsync(cancellationToken);

        await transaction.CommitAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<KnowledgeSnippet>> SearchAsync(string query, int limit, CancellationToken cancellationToken)
    {
        var tokens = TokenRegex
            .Matches(query)
            .Select(m => m.Value.Trim().ToLowerInvariant())
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .Distinct()
            .Take(12)
            .ToList();

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        if (tokens.Count == 0)
        {
            return await GetRecentSnippetsAsync(connection, limit, cancellationToken);
        }

        var ftsQuery = string.Join(" OR ", tokens.Select(t => $"\"{t.Replace("\"", "\"\"")}\""));

        const string sql = """
            SELECT source_url, source_title, content
            FROM knowledge_chunks
            WHERE knowledge_chunks MATCH $query
            ORDER BY bm25(knowledge_chunks)
            LIMIT $limit;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("$query", ftsQuery);
        command.Parameters.AddWithValue("$limit", limit);

        var snippets = new List<KnowledgeSnippet>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            snippets.Add(new KnowledgeSnippet(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2)));
        }

        if (snippets.Count > 0)
        {
            return snippets;
        }

        return await GetRecentSnippetsAsync(connection, limit, cancellationToken);
    }

    public async Task<IReadOnlyList<string>> GetAllChunkContentsAsync(CancellationToken cancellationToken)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            SELECT content
            FROM knowledge_chunks
            ORDER BY rowid DESC;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;

        var chunks = new List<string>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            chunks.Add(reader.GetString(0));
        }

        return chunks;
    }

    private static async Task<IReadOnlyList<KnowledgeSnippet>> GetRecentSnippetsAsync(
        SqliteConnection connection,
        int limit,
        CancellationToken cancellationToken)
    {
        const string sql = """
            SELECT source_url, source_title, content
            FROM knowledge_chunks
            ORDER BY rowid DESC
            LIMIT $limit;
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("$limit", limit);

        var snippets = new List<KnowledgeSnippet>();
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            snippets.Add(new KnowledgeSnippet(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2)));
        }

        return snippets;
    }

    private static KnowledgeSource ReadSource(SqliteDataReader reader)
    {
        DateTimeOffset? indexedAt = reader.IsDBNull(3)
            ? null
            : DateTimeOffset.Parse(reader.GetString(3));
        var contentHash = reader.IsDBNull(4) ? null : reader.GetString(4);
        return new KnowledgeSource(
            reader.GetInt64(0),
            reader.GetString(1),
            reader.GetString(2),
            indexedAt,
            contentHash,
            reader.GetInt64(5) == 1);
    }

    private static string NormalizeUrl(string url)
    {
        if (!Uri.TryCreate(url.Trim(), UriKind.Absolute, out var uri))
        {
            throw new ArgumentException("URL must be absolute.", nameof(url));
        }

        return uri.ToString();
    }
}
