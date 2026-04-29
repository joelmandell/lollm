using System.Diagnostics;
using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ImprovementTrainingService(
    ImprovementCycleService improvementCycleService,
    FeedbackCorpusService feedbackCorpusService,
    EvalFeedbackService evalFeedbackService,
    IHostEnvironment hostEnvironment)
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;
    private TrainingRunState _state = new(false, null, null, null, null, null, false, "No training run started.");

    public async Task<RunImprovementTrainingResponse> RunAsync(
        RunImprovementTrainingRequest request,
        CancellationToken cancellationToken)
    {
        RunImprovementCycleResponse cycle;
        if (request.SkipEval)
        {
            var corpus = await feedbackCorpusService.BuildAsync(
                new BuildFeedbackCorpusRequest(
                    MaxItems: request.FeedbackCorpusMaxItems,
                    LowScoreThreshold: request.FeedbackLowScoreThreshold,
                    IncludePassingSamples: request.IncludePassingSamplesInCorpus,
                    IncludeJsonl: true,
                    IncludeText: true),
                cancellationToken);
            var feedbackStatus = await evalFeedbackService.GetStatusAsync(cancellationToken);
            cycle = new RunImprovementCycleResponse(
                Success: corpus.Success,
                Evaluation: new CodingEvalResponse(0, 0, []),
                FeedbackCorpus: corpus,
                FeedbackStatus: feedbackStatus,
                Message: $"Pre-training eval skipped. Feedback corpus refreshed with {corpus.SelectedItems} samples.");
        }
        else
        {
            cycle = await improvementCycleService.RunAsync(
                new RunImprovementCycleRequest(
                    Prompts: request.Prompts,
                    UseKnowledge: request.UseKnowledge,
                    MaxKnowledgeSnippets: request.MaxKnowledgeSnippets,
                    FeedbackCorpusMaxItems: request.FeedbackCorpusMaxItems,
                    FeedbackLowScoreThreshold: request.FeedbackLowScoreThreshold,
                    IncludePassingSamplesInCorpus: request.IncludePassingSamplesInCorpus),
                cancellationToken);
        }

        await _gate.WaitAsync(cancellationToken);
        try
        {
            ReconcileProcessStateUnsafe();
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
                StopRequested: false,
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
        ReconcileProcessStateUnsafe();
        return new ImprovementTrainingStatusResponse(
            IsRunning: _state.IsRunning,
            ProcessId: _state.ProcessId,
            LogPath: _state.LogPath,
            StartedUtc: _state.StartedUtc,
            CompletedUtc: _state.CompletedUtc,
            ExitCode: _state.ExitCode,
            Message: _state.Message);
    }

    public async Task<ImprovementTrainingStopResponse> StopAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        try
        {
            ReconcileProcessStateUnsafe();
            if (_process is null || _process.HasExited)
            {
                return new ImprovementTrainingStopResponse(
                    Success: true,
                    WasRunning: false,
                    ProcessId: _state.ProcessId,
                    Message: "No active improvement training process was running.");
            }

            var pid = _process.Id;
            _state = _state with { StopRequested = true };

            try
            {
                _process.Kill(entireProcessTree: true);
                await _process.WaitForExitAsync(cancellationToken);
            }
            catch (InvalidOperationException)
            {
                // Process already exited between check and kill.
            }

            _state = _state with
            {
                IsRunning = false,
                CompletedUtc = DateTimeOffset.UtcNow,
                ExitCode = _process.HasExited ? _process.ExitCode : null,
                Message = "Improvement training stopped by request."
            };
            _process = null;

            return new ImprovementTrainingStopResponse(
                Success: true,
                WasRunning: true,
                ProcessId: pid,
                Message: "Improvement training stop signal sent.");
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task<ImprovementTrainingLogResponse> GetLogTailAsync(int tailLines, CancellationToken cancellationToken)
    {
        var boundedTail = Math.Clamp(tailLines, 1, 2000);
        var logPath = _state.LogPath;
        if (string.IsNullOrWhiteSpace(logPath))
        {
            return new ImprovementTrainingLogResponse(
                LogPath: null,
                TailLines: boundedTail,
                Exists: false,
                Lines: [],
                Message: "No training log is available yet.");
        }

        if (!File.Exists(logPath))
        {
            return new ImprovementTrainingLogResponse(
                LogPath: logPath,
                TailLines: boundedTail,
                Exists: false,
                Lines: [],
                Message: "Training log file does not exist.");
        }

        var lines = new List<string>();
        await using var stream = new FileStream(logPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            lines.Add(line);
        }

        var tail = lines.Count > boundedTail ? lines[^boundedTail..] : lines;
        return new ImprovementTrainingLogResponse(
            LogPath: logPath,
            TailLines: boundedTail,
            Exists: true,
            Lines: tail,
            Message: _state.IsRunning ? "Training is running." : "Training is not running.");
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
                Message = _state.StopRequested
                    ? "Improvement training stopped by request."
                    : process.ExitCode == 0 ? "Improvement training completed." : "Improvement training failed."
            };
            if (ReferenceEquals(_process, process))
            {
                _process = null;
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    private void ReconcileProcessStateUnsafe()
    {
        if (_process is null)
        {
            return;
        }

        if (!_process.HasExited)
        {
            if (!_state.IsRunning || _state.ProcessId != _process.Id)
            {
                _state = _state with
                {
                    IsRunning = true,
                    ProcessId = _process.Id,
                    ExitCode = null,
                    CompletedUtc = null,
                    Message = "Improvement training started."
                };
            }

            return;
        }

        _state = _state with
        {
            IsRunning = false,
            CompletedUtc = _state.CompletedUtc ?? DateTimeOffset.UtcNow,
            ExitCode = _process.ExitCode,
            Message = _state.StopRequested
                ? "Improvement training stopped by request."
                : _process.ExitCode == 0 ? "Improvement training completed." : "Improvement training failed."
        };
        _process = null;
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
        bool StopRequested,
        string Message);
}
