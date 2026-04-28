namespace DotNetCodingAgent.Api.Options;

public sealed class LocalModelOptions
{
    public string ModelPath { get; set; } = "local-model.json";
    public int MaxGeneratedTokens { get; set; } = 300;
    public int MaxVocabulary { get; set; } = 50000;
    public int MinTokenLength { get; set; } = 2;
}
