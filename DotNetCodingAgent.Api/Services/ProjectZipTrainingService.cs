using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ProjectZipTrainingService(
    IKnowledgeRepository repository,
    ITrainableLlmClient model)
{
    private static readonly HashSet<string> AllowedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".cs", ".csproj", ".sln", ".props", ".targets", ".json", ".config", ".yml", ".yaml", ".md"
    };

    private const int MaxFileChars = 120_000;
    private const int MaxFiles = 2_500;

    public async Task<ProjectZipTrainingResponse> TrainFromZipAsync(
        Stream zipStream,
        string projectTag,
        int epochs,
        CancellationToken cancellationToken)
    {
        var normalizedTag = ProjectTagHelper.Normalize(projectTag);
        var sourcePrefix = ProjectTagHelper.ToSourcePrefix(normalizedTag);

        using var archive = new ZipArchive(zipStream, ZipArchiveMode.Read, leaveOpen: true);

        var indexedFiles = 0;
        var indexedChunks = 0;

        foreach (var entry in archive.Entries)
        {
            if (indexedFiles >= MaxFiles)
            {
                break;
            }

            if (entry.Length <= 0 || !IsAllowedEntry(entry.FullName))
            {
                continue;
            }

            await using var entryStream = entry.Open();
            using var reader = new StreamReader(entryStream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
            var content = await reader.ReadToEndAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(content))
            {
                continue;
            }

            if (content.Length > MaxFileChars)
            {
                content = content[..MaxFileChars];
            }

            var sourceUrl = $"{sourcePrefix}{entry.FullName.Replace('\\', '/')}";
            var sourceTitle = $"{normalizedTag}: {entry.FullName}";
            var source = await repository.AddSourceAsync(sourceUrl, sourceTitle, cancellationToken);

            var normalizedContent = BuildProjectChunkContent(normalizedTag, entry.FullName, content);
            var chunks = TextChunker.Chunk(normalizedContent);
            indexedChunks += chunks.Count;

            var hash = ComputeHash(normalizedContent);
            await repository.UpdateSourceChunksAsync(
                source.Id,
                sourceTitle,
                hash,
                DateTimeOffset.UtcNow,
                chunks,
                cancellationToken);

            indexedFiles++;
        }

        var projectCorpora = await repository.GetChunkContentsBySourceUrlPrefixAsync(sourcePrefix, cancellationToken);
        if (projectCorpora.Count == 0)
        {
            var snapshot = model.GetSnapshot();
            return new ProjectZipTrainingResponse(
                false,
                normalizedTag,
                indexedFiles,
                indexedChunks,
                0,
                new TrainModelResponse(false, "No trainable project corpus found in zip.", snapshot.TotalTokens, snapshot.VocabularySize, snapshot.LastTrainedUtc),
                "No .NET project files were indexed.");
        }

        await model.TrainAsync(projectCorpora, Math.Max(1, epochs), cancellationToken);
        var trained = model.GetSnapshot();
        var training = new TrainModelResponse(
            true,
            $"Model trained on project '{normalizedTag}' with {projectCorpora.Count} chunks.",
            trained.TotalTokens,
            trained.VocabularySize,
            trained.LastTrainedUtc);

        return new ProjectZipTrainingResponse(
            true,
            normalizedTag,
            indexedFiles,
            indexedChunks,
            projectCorpora.Count,
            training,
            $"Indexed {indexedFiles} files and trained on {projectCorpora.Count} chunks for project '{normalizedTag}'.");
    }

    private static bool IsAllowedEntry(string fullName)
    {
        if (string.IsNullOrWhiteSpace(fullName))
        {
            return false;
        }

        var path = fullName.Replace('\\', '/');
        if (path.Contains("/bin/", StringComparison.OrdinalIgnoreCase) ||
            path.Contains("/obj/", StringComparison.OrdinalIgnoreCase) ||
            path.Contains("/.git/", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var extension = Path.GetExtension(path);
        return AllowedExtensions.Contains(extension);
    }

    private static string BuildProjectChunkContent(string projectTag, string filePath, string content)
    {
        return $"""
            [ProjectTag: {projectTag}]
            [FilePath: {filePath}]
            [ArchitecturalConstraints]
            Reuse existing naming conventions, dependency boundaries, and architectural patterns from this project.
            Avoid introducing breaking architectural changes.
            Support rubberduck reasoning: state assumptions, verify against current code shape, and suggest minimally disruptive changes.

            {content}
            """;
    }

    private static string ComputeHash(string text)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(bytes);
    }
}
