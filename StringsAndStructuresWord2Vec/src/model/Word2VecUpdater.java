package model;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is simple example for model weights update after initial vocab building.
 * If you have built your w2v model, and some time later you've decided that it
 * can be additionally trained over new corpus, here's an example how to do it.
 *
 * PLEASE NOTE: At this moment, no new words will be added to vocabulary/model.
 * Only weights update process will be issued. It's often called "frozen vocab
 * training".
 *
 * @author raver119@gmail.com
 */

public class Word2VecUpdater {

	private static Logger log = LoggerFactory.getLogger(Word2VecUpdater.class);

	public void update(Word2Vec word2Vec, String corpusPath, String outputPath) throws Exception {
		log.info("reading new data...");

		SentenceIterator iterator = new BasicLineIterator(corpusPath);
		TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		word2Vec.setTokenizerFactory(tokenizerFactory);
		word2Vec.setSentenceIterator(iterator);

		log.info("Word2vec uptraining...");

		word2Vec.fit();
		WordVectorSerializer.writeWord2VecModel(word2Vec, outputPath);

	}
}
