package edu.bigdata;

import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;

public class DataCleaning {
    private final String STOP_WORDS_FILE = "stop_words.txt";

    public String preprocess(String input) throws IOException {
        List<String> stopWords = Files.readAllLines(Paths.get(STOP_WORDS_FILE));

        String[] tokens = input.split(" ");
        StringBuilder cleanedInput = new StringBuilder();

        for (String word : tokens) {
            if (!stopWords.contains(word))
                cleanedInput.append(" ").append(word);
        }

        return cleanedInput.toString();
    }

    public Map<Integer, String> tokenize(String input) {
        Tokenizer<Word> tokenizer = PTBTokenizer.factory().getTokenizer(new StringReader(input));

        Map<Integer, String> idToToken = new HashMap<>();
        final int[] id = {0};
        tokenizer.tokenize().forEach(token -> {
            id[0]++;
            idToToken.put(id[0], token.word());
        });
        return idToToken;
    }
}
