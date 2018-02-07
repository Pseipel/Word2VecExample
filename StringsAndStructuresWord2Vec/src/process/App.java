package process;

import java.io.IOException;

import model.Word2VecModelWrapper;
import model.Word2VecRequester;

public class App {

	public static void main(String[] args) {
		Word2VecModelWrapper modelEnglish = new Word2VecModelWrapper("GoogleNews-vectors-negative300.bin.gz");
		Word2VecRequester requester = new Word2VecRequester(modelEnglish.getModel(), 1);

		try {
			requester.requests();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
