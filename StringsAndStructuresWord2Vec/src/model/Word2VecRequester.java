package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Peter
 *
 * Handles requests (on console or from file)
 *
 */
public class Word2VecRequester {

	private Word2Vec vec;
	private int resultListLength;
	private Logger log = LoggerFactory.getLogger(Word2VecModelWrapper.class);

	public Word2VecRequester(Word2Vec vec, int resultListLength) {
		this.vec = vec;
		this.resultListLength = resultListLength;
	}

	public String requestList(String queriesFilePath) throws FileNotFoundException {
		StringBuilder sb = new StringBuilder();
		Scanner scan = new Scanner(new File(queriesFilePath));
		String currentString = "";
		while (scan.hasNextLine()) {
			currentString = scan.nextLine();
			sb.append(resultListLength + " Words closest to '" + currentString + "': "
					+ print(vec.wordsNearestSum(currentString, resultListLength)));
		}
		scan.close();
		return sb.toString();
	}

	public void requests() throws IOException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				log.info("Enter word or sentence (EXIT to break): ");
				String query = br.readLine();
				if (query.equals("EXIT")) {
					System.out.println("END");
					break;
				}
				String response = parseRequest(query);
				log.info(response);
				log.info("END OF LIST");
			}
		}
	}

	public String print(Collection<String> list) {
		String out = "";
		for (String s : list) {
			out += "\n" + s + ": [";
			for (double d : vec.getWordVector(s)) {
				out += " " + d + " ";
			}
			out += "]";
		}
		return out;
	}

	private String parseRequest(String query) {
		StringBuilder sb = new StringBuilder();
		if (query.contains(" ")) {
			List<String> positiveList = new ArrayList<>();
			List<String> negativeList = new ArrayList<>();
			for (String queryPart : query.split(" ")) {
				if (queryPart.startsWith("-")) {
					String negativeQuery = queryPart.replaceFirst("-", "");
					if (vec.getVocab().containsWord(negativeQuery))
						negativeList.add(negativeQuery);
				} else {
					String positiveQuery = queryPart.replaceFirst("\\+", "");
					if (vec.getVocab().containsWord(positiveQuery))
						positiveList.add(positiveQuery);
				}
			}
			sb.append("negatives: " + negativeList + "\n");
			sb.append("positives: " + positiveList + "\n");
			sb.append(print(vec.wordsNearest(positiveList, negativeList, resultListLength)) + "\n");
		} else {
			if (vec.getVocab().containsWord(query)) {
				sb.append(vec.getWordVectorMatrix(query) + "\n");
				sb.append(print(vec.wordsNearest(query, resultListLength)) + "\n");
			} else {
				sb.append(query + " is not contained\n");
			}
		}
		return sb.toString();
	}

	
	// just prototyping: wip

	public void twoModelRequester(Word2Vec vec1, Word2Vec vec2) {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				log.info("Enter word or sentence (EXIT to break): ");
				String query = br.readLine();
				if (query.equals("EXIT")) {
					System.out.println("END");
					break;
				}
				if (vec1.getVocab().containsWord(query)) {
					vec1.getWordVector(query);
					print(vec1.wordsNearest(query, resultListLength), vec1);
					print(vec2.wordsNearest(vec1.getWordVectorMatrix(query), resultListLength), vec2);
				} else {
					log.info(query + " is not contained");
				}
				log.info("END OF LIST");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public String print(Collection<String> list, Word2Vec vec) {
		String out = "";
		for (String s : list) {
			out += "\n" + s + ": [";
			for (double d : vec.getWordVector(s)) {
				out += " " + d + " ";
			}
			out += "]";
		}
		return out;
	}

}
