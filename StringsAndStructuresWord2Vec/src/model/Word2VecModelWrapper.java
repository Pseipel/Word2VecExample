package model;

import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Peter
 *
 * Wrapper to generate, serialize and read models 
 */

public class Word2VecModelWrapper {

	String modelName;
	boolean useCBOW;
	int minWordFrequency;
	int dimensions;
	int windowSize;
	int numberOfIterations;
	String inputFilePath;

	private static Logger log = LoggerFactory.getLogger(Word2VecModelWrapper.class);
	static Word2Vec vec = null;

	public Word2VecModelWrapper(
			String modelName, 
			boolean useCBOW, 
			int dimensions, 
			int minWordFrequency, 
			int windowSize,
			int numberOfIterations, 
			String inputFilePath) {
		this.modelName = modelName;
		this.useCBOW = useCBOW;
		this.dimensions = dimensions;
		this.minWordFrequency = minWordFrequency;
		this.windowSize = windowSize;
		this.numberOfIterations = numberOfIterations;
		this.inputFilePath = inputFilePath;
	}

	public Word2VecModelWrapper(String modelPath) {
		modelName = "data/models/" + modelPath;
		log.info("loading model");
		vec = WordVectorSerializer.readWord2VecModel(modelName);
	}

	public Word2Vec getModel() {
		return vec;
	}

	public Word2Vec generateModel() throws Exception {
		if (useCBOW) {
			modelName += "CBOW.txt";
		} else {
			modelName += "SKIPGRAM.txt";
		}

		log.info("Load & Vectorize Sentences....");
		SentenceIterator iter = new BasicLineIterator(inputFilePath);

		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		log.info("Building model....");
		vec = new Word2Vec.Builder().minWordFrequency(minWordFrequency).iterations(numberOfIterations)
				.layerSize(dimensions).seed(42).windowSize(windowSize).iterate(iter).tokenizerFactory(t).build();

		if (useCBOW) {
			vec.setElementsLearningAlgorithm(new CBOW<VocabWord>());
		} else {
			vec.setElementsLearningAlgorithm(new SkipGram<VocabWord>());
		}

		log.info("Fitting Word2Vec model....");
		vec.fit();

		log.info("Writing word vectors to text file....");
		WordVectorSerializer.writeWord2VecModel(vec, modelName);

		return vec;
	}

}
