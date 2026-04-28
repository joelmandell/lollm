namespace DotNetCodingAgent.Api.Options;

public sealed class LocalModelOptions
{
    public string ModelPath { get; set; } = "local-model.json";
    public int MaxGeneratedTokens { get; set; } = 500;
    public int MaxVocabulary { get; set; } = 250000;
    public int MinTokenLength { get; set; } = 2;
}
