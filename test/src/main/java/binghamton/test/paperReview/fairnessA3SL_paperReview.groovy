package binghamton.test.paperReview;

import binghamton.rl_fairness.A3C.generator_fairness.a3cMDP_fairness;
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate


def fold = 'bias4/'
def outputPath = 'result/paperReview/'+fold;

/*
 * Training
 */
def mdp = new a3cMDP_fairness();

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

