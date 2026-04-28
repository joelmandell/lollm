using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class KnowledgeRefreshWorker(
    IServiceProvider serviceProvider,
    IOptions<KnowledgeOptions> options,
    ILogger<KnowledgeRefreshWorker> logger) : BackgroundService
{
    private readonly TimeSpan _interval = TimeSpan.FromMinutes(Math.Max(5, options.Value.RefreshMinutes));

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);
        using var timer = new PeriodicTimer(_interval);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var scope = serviceProvider.CreateScope();
                var ingestion = scope.ServiceProvider.GetRequiredService<KnowledgeIngestionService>();
                await ingestion.IngestAsync(null, false, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Background knowledge refresh failed.");
            }

            await timer.WaitForNextTickAsync(stoppingToken);
        }
    }
}
