using System.Diagnostics;
using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ImprovementTrainingService(
    ImprovementCycleService improvementCycleService,
    IHostEnvironment hostEnvironment)
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;
    private TrainingRunState _state = new(false, null, null, null, null, null, "No training run started.");

    public async Task<RunImprovementTrainingResponse> RunAsync(
        RunImprovementTrainingRequest request,
        CancellationToken cancellationToken)
    {
        var cycle = await improvementCycleService.RunAsync(
            new RunImprovementCycleRequest(
                Prompts: request.Prompts,
                UseKnowledge: request.UseKnowledge,
                MaxKnowledgeSnippets: request.MaxKnowledgeSnippets,
                FeedbackCorpusMaxItems: request.FeedbackCorpusMaxItems,
                FeedbackLowScoreThreshold: request.FeedbackLowScoreThreshold,
                IncludePassingSamplesInCorpus: request.IncludePassingSamplesInCorpus),
            cancellationToken);

        await _gate.WaitAsync(cancellationToken);
        try
        {
            if (_process is { HasExited: false })
            {
                throw new InvalidOperationException("Improvement training is already running.");
            }

            var repoRoot = Path.GetFullPath(Path.Combine(hostEnvironment.ContentRootPath, ".."));
            var modelStackDir = Path.Combine(repoRoot, "ModelStack");
            var pythonPath = ResolvePythonPath(modelStackDir);
            var logsDir = Path.Combine(modelStackDir, "logs");
            Directory.CreateDirectory(logsDir);
            var logPath = Path.Combine(logsDir, $"quality_cycle_{DateTimeOffset.UtcNow:yyyyMMdd_HHmmss}.log");

            var args = BuildArguments(
                request,
                modelStackDir,
                File.Exists(Path.Combine(modelStackDir, "data", "feedback_corpus.txt")));
            var startInfo = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = args,
                WorkingDirectory = repoRoot,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            var process = Process.Start(startInfo)
                          ?? throw new InvalidOperationException("Failed to start quality cycle process.");
            _process = process;
            _state = new TrainingRunState(
                IsRunning: true,
                ProcessId: process.Id,
                LogPath: logPath,
                StartedUtc: DateTimeOffset.UtcNow,
                CompletedUtc: null,
                ExitCode: null,
                Message: "Improvement training started.");

            _ = Task.Run(() => PumpProcessLogsAsync(process, logPath));

            return new RunImprovementTrainingResponse(
                Success: true,
                Cycle: cycle,
                ProcessId: process.Id,
                Command: $"{pythonPath} {args}",
                LogPath: logPath,
                Message: "Improvement cycle completed and training process started.");
        }
        finally
        {
            _gate.Release();
        }
    }

    public ImprovementTrainingStatusResponse GetStatus()
    {
        return new ImprovementTrainingStatusResponse(
            IsRunning: _state.IsRunning,
            ProcessId: _state.ProcessId,
            LogPath: _state.LogPath,
            StartedUtc: _state.StartedUtc,
            CompletedUtc: _state.CompletedUtc,
            ExitCode: _state.ExitCode,
            Message: _state.Message);
    }

    private async Task PumpProcessLogsAsync(Process process, string logPath)
    {
        await using var stream = new FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.Read);
        await using var writer = new StreamWriter(stream, Encoding.UTF8);

        var stdoutTask = Task.Run(async () =>
        {
            while (await process.StandardOutput.ReadLineAsync() is { } line)
            {
                await writer.WriteLineAsync(line);
                await writer.FlushAsync();
            }
        });

        var stderrTask = Task.Run(async () =>
        {
            while (await process.StandardError.ReadLineAsync() is { } line)
            {
                await writer.WriteLineAsync(line);
                await writer.FlushAsync();
            }
        });

        await Task.WhenAll(stdoutTask, stderrTask);
        await process.WaitForExitAsync();

        await _gate.WaitAsync();
        try
        {
            _state = _state with
            {
                IsRunning = false,
                CompletedUtc = DateTimeOffset.UtcNow,
                ExitCode = process.ExitCode,
                Message = process.ExitCode == 0 ? "Improvement training completed." : "Improvement training failed."
            };
        }
        finally
        {
            _gate.Release();
        }
    }

    private static string BuildArguments(
        RunImprovementTrainingRequest request,
        string modelStackDir,
        bool includeFeedbackCorpus)
    {
        var corpusPath = Path.Combine(modelStackDir, "data", "corpus.txt");
        var canSkipExport = request.SkipExport && File.Exists(corpusPath);
        var parts = new List<string>
        {
            "ModelStack/run_quality_cycle.py",
            "--api-base", "http://localhost:5101",
            "--modelstack-dir", "ModelStack"
        };

        if (canSkipExport)
        {
            parts.Add("--skip-export");
        }

        if (request.SkipEval)
        {
            parts.Add("--skip-eval");
        }

        if (includeFeedbackCorpus)
        {
            parts.Add("--feedback-corpus");
            parts.Add("data/feedback_corpus.txt");
        }

        return string.Join(' ', parts.Select(QuoteArg));
    }

    private static string ResolvePythonPath(string modelStackDir)
    {
        var venvPython = Path.Combine(modelStackDir, ".venv", "bin", "python");
        if (File.Exists(venvPython))
        {
            return venvPython;
        }

        return "python3";
    }

    private static string QuoteArg(string arg)
    {
        if (string.IsNullOrWhiteSpace(arg))
        {
            return "\"\"";
        }

        if (arg.Contains(' ') || arg.Contains('"'))
        {
            return $"\"{arg.Replace("\"", "\\\"", StringComparison.Ordinal)}\"";
        }

        return arg;
    }

    private sealed record TrainingRunState(
        bool IsRunning,
        int? ProcessId,
        string? LogPath,
        DateTimeOffset? StartedUtc,
        DateTimeOffset? CompletedUtc,
        int? ExitCode,
        string Message);
}
