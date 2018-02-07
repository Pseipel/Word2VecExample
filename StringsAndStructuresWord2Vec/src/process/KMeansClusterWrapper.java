package process;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class KMeansClusterWrapper {

	int maxIterations;
	int clusterCountK;
	String outputFileName;
	boolean useEuclidean;

	public KMeansClusterWrapper(String outputFileName, int maxIterations, int clusterCountK, boolean useEuclidean) {
		this.outputFileName = outputFileName;
		this.maxIterations = maxIterations;
		this.clusterCountK = clusterCountK;
		this.useEuclidean = useEuclidean;
	}

	public void process(Word2Vec model) {

		Word2Vec vec = model;
		List<Point> points = new ArrayList<>();
		for (String word : vec.vocab().words()) {
			Point p = new Point(vec.getWordVectorMatrix(word));
			p.setLabel(word);
			points.add(p);
		}

		String distanceFunction = "cosinesimilarity";
		if (useEuclidean)
			distanceFunction = "euclideansimilarity";

		KMeansClustering kmc = KMeansClustering.setup(clusterCountK, maxIterations, distanceFunction);

		ClusterSet clusterSet = kmc.applyTo(points);
		List<Cluster> clusterLst = clusterSet.getClusters();
		FileWriter fw = null;

		try {
			fw = new FileWriter(new File(outputFileName));
			fw.append("########################\n");
			fw.append("maxIterationCount: " + maxIterations + "\n");
			fw.append("clusterCount: " + clusterCountK + "\n");
			fw.append("distanceFunction: " + distanceFunction + "\n");
			fw.append("########################\n");
		} catch (IOException e) {
			e.printStackTrace();
		}

		Cluster c = null;

		for (int i = 0; i < clusterLst.size(); i++) {
			c = clusterLst.get(i);
			Point center = c.getCenter();
			try {
				String centerLabel = c.getId();
				if (center.getLabel() != null) {
					centerLabel += "exact: " + center.getLabel();
				} else {
					centerLabel = "closest: " + "[ ";
					for (String word : vec.wordsNearest(center.getArray(), 10))
						centerLabel += word + " ";
					centerLabel += "]";
				}
				fw.append("\n######\nCenter piont: " + i + ". " + centerLabel + "( " + center.getId()
						+ " )\n######\nCluster size: " + c.getPoints().size() + "\n");
				for (Point p : c.getPoints()) {
					fw.append(p.getLabel() + " (" + c.getId() + ")\n");
				}
			} catch (IOException e) {
				e.printStackTrace();
			}

		}
		System.out.println("Printed " + clusterLst.size() + " clusters");
		System.out.println("END OF PROZESS");
	}
}
