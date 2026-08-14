import Foundation

/// CLIP BPE tokenizer 的 Swift 实现。
/// 词汇表(vocab.json)与合并规则(merges.txt)由转换脚本打包进 App Bundle,
/// 运行时加载,与 Python 端 HuggingFace CLIPTokenizer 行为对齐。
enum CLIPTokenizerSwift {
    private static let startToken = 49406  // <|startoftext|>
    private static let endToken = 49407    // <|endoftext|>
    private static let padToken = 49407

    private struct TokenizerData {
        let vocab: [String: Int]
        let merges: [(String, String)]
        let bpeRanks: [String: Int]
    }

    private static var cache: TokenizerData?
    private static let cacheLock = NSLock()

    // MARK: - 编码

    /// 编码文本为 token id 序列(定长 maxLength,pad 到末尾)
    static func encode(_ text: String, maxLength: Int = 77) -> [Int32] {
        let data = loadData()
        var tokens: [Int32] = [Int32(startToken)]

        // 1. 小写 + 规范化(与 CLIP 预处理一致)
        let normalized = normalize(text)

        // 2. 分词:按空白和标点切分(CLIP 的 basic tokenizer)
        let words = basicTokenize(normalized)

        // 3. 对每个词做 BPE
        for word in words {
            let bpeTokens = bpe(word, data: data)
            for t in bpeTokens {
                if let id = data.vocab[t] {
                    tokens.append(Int32(id))
                    if tokens.count >= maxLength - 1 { break }
                }
            }
            if tokens.count >= maxLength - 1 { break }
        }

        tokens.append(Int32(endToken))

        // 4. 定长填充
        while tokens.count < maxLength {
            tokens.append(Int32(padToken))
        }
        return Array(tokens.prefix(maxLength))
    }

    // MARK: - BPE

    private static func bpe(_ word: String, data: TokenizerData) -> [String] {
        // 字符级初始化
        var parts: [String] = Array(word).map { String($0) }
        parts.append("</w>")

        while parts.count > 1 {
            var bestPair: (String, String)?
            var bestRank = Int.max
            for i in 0..<(parts.count - 1) {
                let pair = (parts[i], parts[i + 1])
                if let rank = data.bpeRanks["\(pair.0) \(pair.1)"] {
                    if rank < bestRank {
                        bestRank = rank
                        bestPair = pair
                    }
                }
            }
            guard let pair = bestPair else { break }

            var merged: [String] = []
            var i = 0
            while i < parts.count {
                if i < parts.count - 1, parts[i] == pair.0, parts[i + 1] == pair.1 {
                    merged.append(pair.0 + pair.1)
                    i += 2
                } else {
                    merged.append(parts[i])
                    i += 1
                }
            }
            parts = merged
        }
        return parts
    }

    // MARK: - 预处理

    private static func normalize(_ text: String) -> String {
        // CLIP:NFKC + 小写 + 去除多余空白
        var s = text.lowercased()
        s = s.applyingTransform(StringTransform("Any-Hex/NFKC"), reverse: false) ?? s
        s = s.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func basicTokenize(_ text: String) -> [String] {
        // 按空白和标点切分(简化版;完整实现需按 CLIP 的正则)
        let pattern = #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}\p{N}]+|[^\s\p{L}\p{N}]+"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            Range(match.range, in: text).map { String(text[$0]) }
        }
    }

    // MARK: - 数据加载

    private static func loadData() -> TokenizerData {
        if let cached = cache { return cached }
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let cached = cache { return cached }

        // 查找 vocab.json/merges.txt:优先 Bundle 根,其次 Models/ 子目录(XcodeGen folder 资源)
        func findResource(_ name: String, _ ext: String) -> URL? {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) { return url }
            if let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Models") { return url }
            return nil
        }

        guard let vocabURL = findResource("vocab", "json"),
              let mergesURL = findResource("merges", "txt"),
              let vocabData = try? Data(contentsOf: vocabURL),
              let vocab = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Int],
              let mergesText = try? String(contentsOf: mergesURL, encoding: .utf8) else {
            // 词汇表缺失时返回退化结果(全 pad),保证不崩溃
            return TokenizerData(vocab: [:], merges: [], bpeRanks: [:])
        }

        let merges: [(String, String)] = mergesText
            .split(separator: "\n")
            .filter { !$0.hasPrefix("#") && $0.contains(" ") }
            .compactMap { line in
                let parts = line.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }

        var ranks: [String: Int] = [:]
        for (i, pair) in merges.enumerated() {
            ranks["\(pair.0) \(pair.1)"] = i
        }

        let data = TokenizerData(vocab: vocab, merges: merges, bpeRanks: ranks)
        cache = data
        return data
    }
}
