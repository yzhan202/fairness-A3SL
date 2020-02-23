package binghamton.test.compasExperiment;


import binghamton.rl.A3C.compas_bias.a3cMDP_compas;
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate


def fold = 0;

def outputPath = 'result/compas_bias/fold'+fold+'/'

/*
 * Training
 */
def mdp = new a3cMDP_compas()

//// Load
//mdp.loadA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser')

mdp.train();

/*
 * Save Result
 */
File outputDir = new File(outputPath)
if (!outputDir.exists()) {
	outputDir.mkdirs();
}
mdp.saveA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser');

println "Finish Training"

