package binghamton.test.paperReview;

import binghamton.rl.A3C.paperReview_bias.a3cMDP_paperReview;
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate


def fold = 'bias1/'
def outputPath = 'result/sensitivePaperReview/'+fold;

/*
 * Training
 */
def mdp = new a3cMDP_paperReview();

// Load
//mdp.loadA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser')

mdp.train();

println "Finish Training"

/*
 * Save Result
 */
File outputDir = new File(outputPath)
if (!outputDir.exists()) {
	outputDir.mkdirs();
}
mdp.saveA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser');



