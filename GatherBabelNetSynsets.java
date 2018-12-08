
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import com.babelscape.util.UniversalPOS;
import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelNetQuery;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetComparator;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.jlt.util.Language;

public class GatherBabelNetSynsets
{
	private static final String FNAME_OUT = "path/to/fileOUT.txt";
	private static final String FNAME_IN = "path/to/fileIN.txt";

	// Simply read a file given the file name
	public static String readFile(String filename) throws IOException
	{
	    String content = null;
	    FileReader reader = null;
	
        reader = new FileReader(new File(filename));
        char[] chars = new char[(int)  new File(filename).length()];
        reader.read(chars);
        content = new String(chars);
        reader.close();
	  
	    return content;
	}
	
	
	// Simply write a file given the file name. It writes the lemma ID and its most frequent sense.
	public static void writeMCS(String fname, ArrayList<String> comp) throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(fname, "UTF-8");
		
		for(String entry : comp) {
			writer.println(entry);
		}
		writer.close();
	}

	/***
	 * Gathers the k most frequent senses expressed as BabelSynsetID objects, for the given lemma.
	 * @param lemma - the lemma for which we need the most frequent sense
	 * @param up - the pos tag of the lemma
	 * @param k - upper-bounds the number of BabelSynsetID objects to return as output
	 * @return
	 */
	public static List<BabelSynsetID> k_mostLikelySenses(String lemma, UniversalPOS up, int k) {
		
		// Get BabelNet singleton instance 
		BabelNet bn = BabelNet.getInstance(); 
		
		// Prepare query in English for the specified lemma, POS tag
		BabelNetQuery query = new BabelNetQuery.Builder(lemma)
							.from(Language.EN)
							.POS(up)
							.build();
		
		// Get the associated BabelSynset objects to the query and sort them for frequency 
		List<BabelSynset> byl = bn.getSynsets(query).stream()
				.sorted(new BabelSynsetComparator(lemma, Language.EN))
				.collect(Collectors.toList());

		// The list of BabelSynsetID objects sorted by frequency
		List<BabelSynsetID> bylIDs = new ArrayList<BabelSynsetID>();
		
		for(BabelSynset b : byl) 
			bylIDs.add(b.getID());
		
		// Truncate the k most frequent BabelSynsets
		if (k < bylIDs.size())  bylIDs = bylIDs.subList(0, k);
		else bylIDs = bylIDs.subList(0, bylIDs.size());
		
		return bylIDs;
	}
	
	/***
	 * Creates the annotation wordid|most_frequent_sense for each word in the input document.
	 * @throws IOException
	 */
	public static void annotate() throws IOException {
		String tt = readFile(FNAME_IN);
		
		// It contains all the lines of the file FNAME_IN as a list [[wordid, lemma, POS],...]
		ArrayList<ArrayList<String>> ambs = new ArrayList<ArrayList<String>>();
		
		// Analyze the input file: wordid|lemma|POS|\n
		for (String ss : tt.split("\n")) {
			ArrayList<String> comps = new ArrayList<String>(Arrays.asList(ss.split("\\|")));
			ambs.add(comps);
		}
		
		// It contains the associations wordid|most_frequent_sense
		ArrayList<String> comps = new ArrayList<String>();
		for (ArrayList<String> entry : ambs) {
			
			// Get the most frequent sense of entry.get(1) having POS entry.get(2)
			List<BabelSynsetID> MFS = k_mostLikelySenses(entry.get(1), UniversalPOS.valueOf(entry.get(2)), 1);
			
			// Use sense - when there's no MFS
			String sense = MFS.size() == 1 ? MFS.get(0).toString() : "-";
			
			// Generates the line of the output file
			String out = entry.get(0) + "|" + sense;
			comps.add(out);
		}
		// Go write the output file
		writeMCS(FNAME_OUT, comps);
	}
	
	
	public static void main(String[] args) throws IOException
	{
		GatherBabelNetSynsets.annotate();
	}
}
