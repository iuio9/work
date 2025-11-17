<template>
  <div class="evaluation-metrics">
    <h2>数据集模型评估图表计算说明</h2>

    <section>
      <h3>1. 准确率 (Accuracy)</h3>
      <p>
        准确率是最直观的理解模型性能的方法之一，它表示模型预测正确的比例。
        <br />
        公式为：
      </p>
      <div class="formula">
        Accuracy =
        <span class="fraction">
          <span class="numerator">True Positives + True Negatives</span>
          <span class="denominator">Total Predictions</span>
        </span>
      </div>
      <p>
        但是，在不平衡的数据集中，高准确率并不一定意味着好的模型性能。
      </p>
    </section>

    <section>
      <h3>2. 精确率 (Precision) 和 召回率 (Recall)</h3>
      <p>
        - <strong>精确率</strong>：所有被预测为正类的样本中，实际为正类的比例。
        <br />
        公式为：
      </p>
      <div class="formula">
        Precision =
        <span class="fraction">
          <span class="numerator">True Positives</span>
          <span class="denominator">True Positives + False Positives</span>
        </span>
      </div>
      <p>
        - <strong>召回率</strong>：所有实际为正类的样本中，被正确预测为正类的比例。
        <br />
        公式为：
      </p>
      <div class="formula">
        Recall =
        <span class="fraction">
          <span class="numerator">True Positives</span>
          <span class="denominator">True Positives + False Negatives</span>
        </span>
      </div>
    </section>

    <section>
      <h3>3. F1 分数 (F1 Score)</h3>
      <p>
        F1 分数是精确率和召回率的调和平均值，用于平衡两者之间的关系。公式为：
      </p>
      <div class="formula">
        F1 Score = 2 ×
        <span class="fraction">
          <span class="numerator">Precision × Recall</span>
          <span class="denominator">Precision + Recall</span>
        </span>
      </div>
    </section>

    <section>
      <h3>4. AUC - ROC 曲线下的面积 (Area Under the Curve - Receiver Operating Characteristic)</h3>
      <p>
        AUC-ROC 是一个衡量分类器整体性能的指标，特别是对于二元分类问题。ROC 曲线展示了不同阈值下真阳率（TPR）和假阳率（FPR）之间的权衡。AUC 值越接近于1，模型的区分能力越强。
      </p>
    </section>

    <section>
      <h3>5. BLEU, ROUGE, METEOR</h3>
      <p>
        这些指标主要用于文本生成任务，如机器翻译、摘要生成等，它们通过比较生成的文本与参考文本之间的相似性来评估模型性能。
      </p>
      <ul>
        <li><strong>BLEU (Bilingual Evaluation Understudy)</strong>: 衡量生成句子与参考句子之间 n-gram 的重合度。</li>
        <li><strong>ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</strong>: 主要关注召回率，包括 ROUGE-N（基于 n-gram）、ROUGE-L（最长公共子序列）等。</li>
        <li><strong>METEOR (Metric for Evaluation of Translation with Explicit ORdering)</strong>: 结合了精确率、召回率和其他因素，提供更全面的评估。</li>
      </ul>
    </section>
  </div>
</template>

<script setup>
// 如果需要定义任何逻辑或响应式数据，可以在这里添加
</script>

<style scoped>
.evaluation-metrics {
  font-family: Arial, sans-serif;
}

h2, h3 {
  color: #2c3e50;
}

.formula {
  display: inline-block;
  margin: 10px 0;
  padding: 5px;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.fraction {
  display: inline-block;
  text-align: center;
}

.numerator, .denominator {
  display: block;
}

.numerator {
  border-bottom: 1px solid #000;
  padding-bottom: 2px;
}

.denominator {
  padding-top: 2px;
}

ul {
  list-style-type: disc;
  margin-left: 20px;
}
</style>
