namespace DotNetCodingAgent.Api.Options;

public sealed class ModelBackendOptions
{
    public string Provider { get; set; } = "hybrid";
    public string TransformerBaseUrl { get; set; } = "http://127.0.0.1:8010";
    public int TransformerTimeoutSeconds { get; set; } = 120;
    public string ExportDataDirectory { get; set; } = "../ModelStack/data";
}
