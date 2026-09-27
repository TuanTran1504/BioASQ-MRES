package evaluation;

import data.Question;
import data.Task1bData;

/**
 * Thin output adapter around the official BioASQ evaluator classes.
 *
 * The official CLI prints corpus aggregates only. This class calls the same
 * official QuestionAnswerEvaluator implementation and exposes its per-question
 * values as tab-separated records for error analysis. It contains no metric
 * formulas.
 */
public final class BioASQPerQuestionEvaluator {
    private BioASQPerQuestionEvaluator() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException("Usage: GOLD_JSON PREDICTION_JSON CHALLENGE_VERSION");
        }
        int version = Integer.parseInt(args[2]);
        Task1bData goldData = new Task1bData(version, true);
        Task1bData systemData = new Task1bData(version, false);
        goldData.readData(args[0]);
        systemData.readData(args[1]);

        for (int i = 0; i < goldData.numQuestions(); i++) {
            Question gold = goldData.getQuestion(i);
            Question response = systemData.getQuestion(gold.getId());
            QuestionAnswerEvaluator evaluator = new QuestionAnswerEvaluator(
                    gold.getId(), gold.getType(), version);
            if (response != null) {
                evaluator.calculatePhaseBMeasuresForPair(gold, response);
            }
            System.out.println(
                    gold.getId() + "\t" + gold.getType() + "\t"
                    + evaluator.getAccuracyYesNo() + "\t"
                    + evaluator.getStrictAccuracy() + "\t"
                    + evaluator.getLenientAccuracy() + "\t"
                    + evaluator.getMRR() + "\t"
                    + evaluator.getPrecisionEA() + "\t"
                    + evaluator.getRecallEA() + "\t"
                    + evaluator.getF1EA());
        }
    }
}
