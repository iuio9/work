<template>
  <div class="model-distillation-container">
    <n-card title="大小模型协同训练平台" :bordered="false" class="h-full rounded-8px shadow-sm">
      <!-- 顶部统计卡片 -->
      <n-space vertical :size="16" class="mb-4">
        <n-grid :cols="5" :x-gap="16">
          <n-gi>
            <n-statistic label="总任务数" :value="taskStats.total">
              <template #prefix>
                <n-icon :component="LayersOutline" color="#409EFF" />
              </template>
            </n-statistic>
          </n-gi>
          <n-gi>
            <n-statistic label="运行中" :value="taskStats.running">
              <template #prefix>
                <n-icon :component="PlayCircleOutline" color="#67C23A" />
              </template>
            </n-statistic>
          </n-gi>
          <n-gi>
            <n-statistic label="已完成" :value="taskStats.completed">
              <template #prefix>
                <n-icon :component="CheckmarkCircleOutline" color="#909399" />
              </template>
            </n-statistic>
          </n-gi>
          <n-gi>
            <n-statistic label="平均准确率" :value="taskStats.avgAccuracy" suffix="%">
              <template #prefix>
                <n-icon :component="TrendingUpOutline" color="#E6A23C" />
              </template>
            </n-statistic>
          </n-gi>
          <n-gi>
            <n-statistic label="GPU使用率" :value="taskStats.gpuUsage" suffix="%">
              <template #prefix>
                <n-icon :component="HardwareChipOutline" color="#F56C6C" />
              </template>
            </n-statistic>
          </n-gi>
        </n-grid>
      </n-space>

      <!-- 主体标签页 -->
      <n-tabs v-model:value="activeTab" type="line" animated>
        <!-- 1. 模型配置标签页 -->
        <n-tab-pane name="model-config" tab="模型配置">
          <div class="p-4">
            <n-space vertical :size="24">
              <!-- 教师模型配置 -->
              <n-card title="教师模型（大模型）" :bordered="false" size="small" hoverable>
                <n-form :label-width="120">
                  <n-grid :cols="2" :x-gap="24">
                    <n-gi>
                      <n-form-item label="选择教师模型">
                        <n-select
                          v-model:value="teacherModel.modelId"
                          :options="teacherModelOptions"
                          placeholder="选择预训练大模型"
                          @update:value="handleTeacherModelChange"
                        >
                          <template #label="{ option }">
                            <div class="flex items-center">
                              <n-icon :component="SchoolOutline" class="mr-2" />
                              {{ option.label }}
                            </div>
                          </template>
                        </n-select>
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="模型参数量">
                        <n-input v-model:value="teacherModel.paramSize" placeholder="例如：7B, 13B, 70B" disabled />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="模型路径">
                        <n-input v-model:value="teacherModel.modelPath" placeholder="模型文件路径或HuggingFace ID" />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="量化方式">
                        <n-select
                          v-model:value="teacherModel.quantization"
                          :options="quantizationOptions"
                          placeholder="选择量化方式"
                        />
                      </n-form-item>
                    </n-gi>
                  </n-grid>
                </n-form>
              </n-card>

              <!-- 学生模型配置 -->
              <n-card title="学生模型（小模型）" :bordered="false" size="small" hoverable>
                <n-form :label-width="120">
                  <n-grid :cols="2" :x-gap="24">
                    <n-gi>
                      <n-form-item label="选择学生模型">
                        <n-select
                          v-model:value="studentModel.modelId"
                          :options="studentModelOptions"
                          placeholder="选择小模型架构"
                          @update:value="handleStudentModelChange"
                        >
                          <template #label="{ option }">
                            <div class="flex items-center">
                              <n-icon :component="PersonOutline" class="mr-2" />
                              {{ option.label }}
                            </div>
                          </template>
                        </n-select>
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="模型参数量">
                        <n-input v-model:value="studentModel.paramSize" placeholder="例如：110M, 350M, 1.5B" disabled />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="初始化方式">
                        <n-select
                          v-model:value="studentModel.initMethod"
                          :options="initMethodOptions"
                          placeholder="选择初始化方式"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="预训练权重">
                        <n-input v-model:value="studentModel.pretrainPath" placeholder="可选：学生模型预训练权重" />
                      </n-form-item>
                    </n-gi>
                  </n-grid>
                </n-form>
              </n-card>

              <!-- LoRA 配置 -->
              <n-card title="LoRA 高效微调配置" :bordered="false" size="small" hoverable>
                <template #header-extra>
                  <n-tag type="success" size="small">
                    <template #icon>
                      <n-icon :component="FlashOutline" />
                    </template>
                    参数高效微调
                  </n-tag>
                </template>
                <n-form :label-width="140">
                  <n-grid :cols="2" :x-gap="24">
                    <n-gi>
                      <n-form-item label="LoRA Rank" required>
                        <n-input-number
                          v-model:value="loraConfig.rank"
                          :min="8"
                          :max="256"
                          :step="8"
                          placeholder="建议 8-64"
                          style="width: 100%"
                        >
                          <template #suffix>
                            <n-tooltip trigger="hover">
                              <template #trigger>
                                <n-icon :component="InformationCircleOutline" />
                              </template>
                              Rank 值越大，模型表达能力越强，但参数量增加。建议范围：8-64
                            </n-tooltip>
                          </template>
                        </n-input-number>
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="LoRA Alpha">
                        <n-input-number
                          v-model:value="loraConfig.alpha"
                          :min="1"
                          :max="128"
                          placeholder="通常设置为 rank 的 2 倍"
                          style="width: 100%"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="LoRA Dropout">
                        <n-input-number
                          v-model:value="loraConfig.dropout"
                          :min="0"
                          :max="1"
                          :step="0.05"
                          placeholder="防止过拟合"
                          style="width: 100%"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="目标模块">
                        <n-select
                          v-model:value="loraConfig.targetModules"
                          multiple
                          :options="loraTargetOptions"
                          placeholder="选择要应用 LoRA 的模块"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="LoRA 应用层">
                        <n-input
                          v-model:value="loraConfig.layers"
                          placeholder="例如：all 或 0-11"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="Bias 训练">
                        <n-select
                          v-model:value="loraConfig.biasTrain"
                          :options="biasTrainOptions"
                          placeholder="选择 bias 参数训练策略"
                        />
                      </n-form-item>
                    </n-gi>
                  </n-grid>
                </n-form>
              </n-card>

              <!-- 知识蒸馏配置 -->
              <n-card title="知识蒸馏超参数" :bordered="false" size="small" hoverable>
                <n-form :label-width="140">
                  <n-grid :cols="3" :x-gap="24">
                    <n-gi>
                      <n-form-item label="蒸馏温度 (T)">
                        <n-input-number
                          v-model:value="distillConfig.temperature"
                          :min="1"
                          :max="20"
                          :step="0.5"
                          placeholder="软标签平滑度"
                          style="width: 100%"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="硬标签权重 (α)">
                        <n-input-number
                          v-model:value="distillConfig.hardLabelWeight"
                          :min="0"
                          :max="1"
                          :step="0.1"
                          placeholder="真实标签损失权重"
                          style="width: 100%"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="软标签权重 (β)">
                        <n-input-number
                          v-model:value="distillConfig.softLabelWeight"
                          :min="0"
                          :max="1"
                          :step="0.1"
                          placeholder="教师输出损失权重"
                          style="width: 100%"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="蒸馏损失类型">
                        <n-select
                          v-model:value="distillConfig.lossType"
                          :options="distillLossOptions"
                          placeholder="选择损失函数"
                        />
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="中间层蒸馏">
                        <n-switch v-model:value="distillConfig.intermediateLayers">
                          <template #checked>启用</template>
                          <template #unchecked>禁用</template>
                        </n-switch>
                      </n-form-item>
                    </n-gi>
                    <n-gi>
                      <n-form-item label="注意力蒸馏">
                        <n-switch v-model:value="distillConfig.attentionDistill">
                          <template #checked>启用</template>
                          <template #unchecked>禁用</template>
                        </n-switch>
                      </n-form-item>
                    </n-gi>
                  </n-grid>
                </n-form>
              </n-card>

              <!-- 操作按钮 -->
              <n-space justify="end">
                <n-button @click="handleResetConfig">
                  <template #icon>
                    <n-icon :component="RefreshOutline" />
                  </template>
                  重置配置
                </n-button>
                <n-button type="primary" @click="handleSaveConfig">
                  <template #icon>
                    <n-icon :component="SaveOutline" />
                  </template>
                  保存配置
                </n-button>
              </n-space>
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 2. 训练任务标签页 -->
        <n-tab-pane name="training-tasks" tab="训练任务">
          <div class="p-4">
            <n-space vertical :size="16">
              <!-- 操作栏 -->
              <n-space>
                <n-button type="primary" @click="showCreateTaskModal = true">
                  <template #icon>
                    <n-icon :component="AddCircleOutline" />
                  </template>
                  创建训练任务
                </n-button>
                <n-button @click="refreshTasks">
                  <template #icon>
                    <n-icon :component="RefreshOutline" />
                  </template>
                  刷新
                </n-button>
                <n-input
                  v-model:value="taskSearchKeyword"
                  placeholder="搜索任务名称或ID"
                  style="width: 300px"
                >
                  <template #prefix>
                    <n-icon :component="SearchOutline" />
                  </template>
                </n-input>
              </n-space>

              <!-- 任务列表 -->
              <n-data-table
                :columns="taskColumns"
                :data="filteredTasks"
                :loading="tasksLoading"
                :bordered="false"
                :pagination="taskPagination"
                :row-key="(row) => row.taskId"
              />
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 3. 训练监控标签页 -->
        <n-tab-pane name="training-monitor" tab="训练监控">
          <div class="p-4">
            <n-empty v-if="!selectedTask" description="请从任务列表中选择一个训练任务进行监控">
              <template #extra>
                <n-button size="small" @click="activeTab = 'training-tasks'">
                  前往任务列表
                </n-button>
              </template>
            </n-empty>

            <div v-else>
              <n-space vertical :size="24">
                <!-- 任务基本信息 -->
                <n-card title="任务信息" :bordered="false">
                  <n-descriptions :column="4" bordered>
                    <n-descriptions-item label="任务ID">
                      {{ selectedTask.taskId }}
                    </n-descriptions-item>
                    <n-descriptions-item label="任务名称">
                      {{ selectedTask.taskName }}
                    </n-descriptions-item>
                    <n-descriptions-item label="任务状态">
                      <n-tag :type="getStatusType(selectedTask.status)">
                        {{ selectedTask.status }}
                      </n-tag>
                    </n-descriptions-item>
                    <n-descriptions-item label="当前轮次">
                      {{ selectedTask.currentEpoch }} / {{ selectedTask.totalEpochs }}
                    </n-descriptions-item>
                    <n-descriptions-item label="教师模型">
                      {{ selectedTask.teacherModel }}
                    </n-descriptions-item>
                    <n-descriptions-item label="学生模型">
                      {{ selectedTask.studentModel }}
                    </n-descriptions-item>
                    <n-descriptions-item label="LoRA Rank">
                      {{ selectedTask.loraRank }}
                    </n-descriptions-item>
                    <n-descriptions-item label="蒸馏温度">
                      {{ selectedTask.temperature }}
                    </n-descriptions-item>
                  </n-descriptions>
                </n-card>

                <!-- 训练进度 -->
                <n-card title="训练进度" :bordered="false">
                  <n-space vertical :size="16">
                    <div>
                      <div class="flex justify-between mb-2">
                        <span>整体进度</span>
                        <span>{{ selectedTask.progress }}%</span>
                      </div>
                      <n-progress
                        type="line"
                        :percentage="selectedTask.progress"
                        :indicator-placement="'inside'"
                        processing
                      />
                    </div>
                    <n-grid :cols="4" :x-gap="16">
                      <n-gi>
                        <n-statistic label="已训练样本" :value="selectedTask.trainedSamples" />
                      </n-gi>
                      <n-gi>
                        <n-statistic label="预计剩余时间" :value="selectedTask.estimatedTime" />
                      </n-gi>
                      <n-gi>
                        <n-statistic label="当前学习率" :value="selectedTask.learningRate" />
                      </n-gi>
                      <n-gi>
                        <n-statistic label="显存占用" :value="selectedTask.memoryUsed" suffix="GB" />
                      </n-gi>
                    </n-grid>
                  </n-space>
                </n-card>

                <!-- 损失曲线 -->
                <n-card title="损失曲线" :bordered="false">
                  <div ref="lossChartRef" style="width: 100%; height: 400px"></div>
                </n-card>

                <!-- 准确率曲线 -->
                <n-card title="准确率对比" :bordered="false">
                  <div ref="accuracyChartRef" style="width: 100%; height: 400px"></div>
                </n-card>

                <!-- 资源监控 -->
                <n-grid :cols="2" :x-gap="16">
                  <n-gi>
                    <n-card title="GPU 使用率" :bordered="false">
                      <div ref="gpuChartRef" style="width: 100%; height: 300px"></div>
                    </n-card>
                  </n-gi>
                  <n-gi>
                    <n-card title="显存使用" :bordered="false">
                      <div ref="memoryChartRef" style="width: 100%; height: 300px"></div>
                    </n-card>
                  </n-gi>
                </n-grid>
              </n-space>
            </div>
          </div>
        </n-tab-pane>

        <!-- 4. LoRA 配置管理标签页 -->
        <n-tab-pane name="lora-management" tab="LoRA 配置管理">
          <div class="p-4">
            <n-space vertical :size="16">
              <!-- 操作栏 -->
              <n-space>
                <n-button type="primary" @click="showSaveLoraPresetModal = true">
                  <template #icon>
                    <n-icon :component="SaveOutline" />
                  </template>
                  保存为预设
                </n-button>
                <n-button @click="refreshLoraPresets">
                  <template #icon>
                    <n-icon :component="RefreshOutline" />
                  </template>
                  刷新
                </n-button>
              </n-space>

              <!-- LoRA 预设列表 -->
              <n-data-table
                :columns="loraPresetColumns"
                :data="loraPresets"
                :loading="loraPresetsLoading"
                :bordered="false"
              />
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 5. 已训练模型列表标签页 -->
        <n-tab-pane name="trained-models" tab="已训练模型">
          <div class="p-4">
            <n-space vertical :size="16">
              <n-alert type="info" title="说明" closable>
                这里展示所有已完成训练的大小模型协同训练任务。您可以在"数据标注 > 自动标注"页面中使用这些模型进行图像标注。
              </n-alert>

              <!-- 搜索和筛选 -->
              <n-card title="筛选条件" :bordered="false" size="small">
                <n-form inline :label-width="80">
                  <n-form-item label="任务名称">
                    <n-input
                      v-model:value="trainedModelsSearch.taskName"
                      placeholder="输入任务名称"
                      clearable
                      style="width: 200px"
                    />
                  </n-form-item>
                  <n-form-item label="教师模型">
                    <n-select
                      v-model:value="trainedModelsSearch.teacherModel"
                      :options="teacherModelOptions.map(m => ({ label: m.label, value: m.value }))"
                      placeholder="选择教师模型"
                      clearable
                      style="width: 200px"
                    />
                  </n-form-item>
                  <n-form-item label="学生模型">
                    <n-select
                      v-model:value="trainedModelsSearch.studentModel"
                      :options="studentModelOptions.map(m => ({ label: m.label, value: m.value }))"
                      placeholder="选择学生模型"
                      clearable
                      style="width: 200px"
                    />
                  </n-form-item>
                  <n-form-item label="最小准确率">
                    <n-input-number
                      v-model:value="trainedModelsSearch.minAccuracy"
                      :min="0"
                      :max="100"
                      placeholder="最小准确率"
                      style="width: 150px"
                    >
                      <template #suffix>%</template>
                    </n-input-number>
                  </n-form-item>
                  <n-form-item>
                    <n-space>
                      <n-button @click="resetTrainedModelsSearch">
                        <template #icon>
                          <n-icon :component="RefreshOutline" />
                        </template>
                        重置
                      </n-button>
                      <n-button type="primary" @click="refreshTasks">
                        <template #icon>
                          <n-icon :component="SearchOutline" />
                        </template>
                        搜索
                      </n-button>
                    </n-space>
                  </n-form-item>
                </n-form>
              </n-card>

              <!-- 已训练模型列表 -->
              <n-card title="已训练模型列表" :bordered="false">
                <template #header-extra>
                  <n-space>
                    <n-tag type="success">
                      共 {{ filteredCompletedModels.length }} 个可用模型
                    </n-tag>
                    <n-button size="small" @click="refreshTasks">
                      <template #icon>
                        <n-icon :component="RefreshOutline" />
                      </template>
                      刷新
                    </n-button>
                  </n-space>
                </template>

                <n-data-table
                  :columns="trainedModelsColumns"
                  :data="filteredCompletedModels"
                  :loading="tasksLoading"
                  :pagination="trainedModelsPagination"
                  :bordered="false"
                />
              </n-card>
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 6. 推理任务列表（自动标注） -->
        <n-tab-pane name="inference-tasks" tab="推理任务（自动标注）">
          <div class="p-4">
            <n-space vertical :size="16">
              <div class="flex justify-between items-center">
                <n-space>
                  <n-icon :component="InformationCircleOutline" size="20" color="#409EFF" />
                  <span class="text-base font-medium">推理任务列表 - 使用训练好的模型进行自动标注</span>
                </n-space>
                <n-button type="primary" @click="refreshInferenceTasks">
                  <template #icon>
                    <n-icon :component="RefreshOutline" />
                  </template>
                  刷新列表
                </n-button>
              </div>

              <n-card title="推理任务" :bordered="false" hoverable>
                <n-data-table
                  :columns="inferenceColumns"
                  :data="inferenceTasksData"
                  :loading="inferenceTasksLoading"
                  :scroll-x="1800"
                  :pagination="{ pageSize: 10 }"
                  :bordered="false"
                />
              </n-card>
            </n-space>
          </div>
        </n-tab-pane>
      </n-tabs>
    </n-card>

    <!-- 创建训练任务对话框 -->
    <n-modal
      v-model:show="showCreateTaskModal"
      preset="card"
      title="创建训练任务"
      style="width: 800px"
      :bordered="false"
    >
      <n-form ref="taskFormRef" :model="taskForm" :rules="taskFormRules" :label-width="140">
        <n-form-item label="任务名称" path="taskName">
          <n-input v-model:value="taskForm.taskName" placeholder="输入任务名称" />
        </n-form-item>

        <n-form-item label="任务描述" path="description">
          <n-input
            v-model:value="taskForm.description"
            type="textarea"
            placeholder="描述训练任务目标和用途"
            :rows="3"
          />
        </n-form-item>

        <n-form-item label="训练数据集" path="datasetId">
          <n-select
            v-model:value="taskForm.datasetId"
            :options="datasetOptions"
            placeholder="选择训练数据集"
          />
        </n-form-item>

        <n-form-item label="验证数据集" path="valDatasetId">
          <n-select
            v-model:value="taskForm.valDatasetId"
            :options="datasetOptions"
            placeholder="选择验证数据集"
          />
        </n-form-item>

        <n-divider>训练超参数</n-divider>

        <n-grid :cols="2" :x-gap="24">
          <n-gi>
            <n-form-item label="训练轮数 (Epochs)" path="epochs">
              <n-input-number
                v-model:value="taskForm.epochs"
                :min="1"
                :max="100"
                placeholder="训练轮数"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
          <n-gi>
            <n-form-item label="批次大小 (Batch)" path="batchSize">
              <n-input-number
                v-model:value="taskForm.batchSize"
                :min="1"
                :max="512"
                placeholder="批次大小"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
          <n-gi>
            <n-form-item label="学习率" path="learningRate">
              <n-input-number
                v-model:value="taskForm.learningRate"
                :min="0.00001"
                :max="0.1"
                :step="0.00001"
                placeholder="学习率"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
          <n-gi>
            <n-form-item label="权重衰减" path="weightDecay">
              <n-input-number
                v-model:value="taskForm.weightDecay"
                :min="0"
                :max="0.1"
                :step="0.0001"
                placeholder="L2正则化"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
          <n-gi>
            <n-form-item label="梯度累积步数" path="gradAccumSteps">
              <n-input-number
                v-model:value="taskForm.gradAccumSteps"
                :min="1"
                :max="32"
                placeholder="梯度累积"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
          <n-gi>
            <n-form-item label="最大梯度范数" path="maxGradNorm">
              <n-input-number
                v-model:value="taskForm.maxGradNorm"
                :min="0.1"
                :max="10"
                :step="0.1"
                placeholder="梯度裁剪"
                style="width: 100%"
              />
            </n-form-item>
          </n-gi>
        </n-grid>

        <n-form-item label="学习率调度器" path="lrScheduler">
          <n-select
            v-model:value="taskForm.lrScheduler"
            :options="lrSchedulerOptions"
            placeholder="选择学习率调度策略"
          />
        </n-form-item>

        <n-form-item label="优化器" path="optimizer">
          <n-select
            v-model:value="taskForm.optimizer"
            :options="optimizerOptions"
            placeholder="选择优化器"
          />
        </n-form-item>

        <n-form-item label="GPU 设备" path="gpuDevices">
          <n-select
            v-model:value="taskForm.gpuDevices"
            multiple
            :options="gpuOptions"
            placeholder="选择训练使用的GPU"
          />
        </n-form-item>

        <n-form-item label="自动保存检查点">
          <n-switch v-model:value="taskForm.autoSaveCheckpoint">
            <template #checked>每隔 {{ taskForm.checkpointInterval }} 轮保存</template>
            <template #unchecked>不自动保存</template>
          </n-switch>
        </n-form-item>

        <n-form-item v-if="taskForm.autoSaveCheckpoint" label="保存间隔" path="checkpointInterval">
          <n-input-number
            v-model:value="taskForm.checkpointInterval"
            :min="1"
            :max="50"
            placeholder="每N轮保存一次"
            style="width: 100%"
          />
        </n-form-item>

        <!-- Qwen2.5-VL多模型配置 -->
        <template v-if="isQwenTeacher">
          <n-divider>Qwen2.5-VL多模型配置</n-divider>

          <n-grid :cols="2" :x-gap="24">
            <n-gi>
              <n-form-item label="任务类型" path="taskType">
                <n-select
                  v-model:value="taskForm.taskType"
                  :options="taskTypeOptions"
                  placeholder="选择任务类型"
                />
              </n-form-item>
            </n-gi>
            <n-gi>
              <n-form-item label="学生模型大小" path="studentModelSize">
                <n-select
                  v-model:value="taskForm.studentModelSize"
                  :options="studentModelSizeOptions"
                  placeholder="选择模型大小"
                />
              </n-form-item>
            </n-gi>
            <n-gi>
              <n-form-item label="分类类别数" path="numClasses">
                <n-input-number
                  v-model:value="taskForm.numClasses"
                  :min="2"
                  :max="1000"
                  placeholder="类别数"
                  style="width: 100%"
                />
              </n-form-item>
            </n-gi>
            <n-gi>
              <n-form-item label="图像尺寸" path="imageSize">
                <n-input-number
                  v-model:value="taskForm.imageSize"
                  :min="32"
                  :max="1024"
                  :step="32"
                  placeholder="输入图像大小"
                  style="width: 100%"
                />
              </n-form-item>
            </n-gi>
            <n-gi>
              <n-form-item label="蒸馏类型" path="distillationType">
                <n-select
                  v-model:value="taskForm.distillationType"
                  :options="distillationTypeOptions"
                  placeholder="选择蒸馏策略"
                />
              </n-form-item>
            </n-gi>
            <n-gi>
              <n-form-item label="特征损失类型" path="featureLossType">
                <n-select
                  v-model:value="taskForm.featureLossType"
                  :options="featureLossTypeOptions"
                  placeholder="选择损失函数"
                />
              </n-form-item>
            </n-gi>
          </n-grid>

          <n-form-item label="启用特征对齐">
            <n-switch v-model:value="taskForm.alignFeature">
              <template #checked>启用</template>
              <template #unchecked>禁用</template>
            </n-switch>
          </n-form-item>
        </template>
      </n-form>

      <template #footer>
        <n-space justify="end">
          <n-button @click="showCreateTaskModal = false">取消</n-button>
          <n-button type="primary" @click="handleCreateTask" :loading="creatingTask">
            创建任务
          </n-button>
        </n-space>
      </template>
    </n-modal>

    <!-- 保存 LoRA 预设对话框 -->
    <n-modal
      v-model:show="showSaveLoraPresetModal"
      preset="card"
      title="保存 LoRA 预设"
      style="width: 500px"
      :bordered="false"
    >
      <n-form :label-width="100">
        <n-form-item label="预设名称">
          <n-input v-model:value="loraPresetName" placeholder="为这组 LoRA 配置起个名字" />
        </n-form-item>
        <n-form-item label="预设描述">
          <n-input
            v-model:value="loraPresetDesc"
            type="textarea"
            placeholder="描述这组配置的特点和适用场景"
            :rows="3"
          />
        </n-form-item>
      </n-form>

      <template #footer>
        <n-space justify="end">
          <n-button @click="showSaveLoraPresetModal = false">取消</n-button>
          <n-button type="primary" @click="handleSaveLoraPreset" :loading="savingLoraPreset">
            保存
          </n-button>
        </n-space>
      </template>
    </n-modal>

    <!-- 推理对话框 -->
    <InferenceDialog
      v-model:show="showInferenceDialog"
      :task="selectedTaskForInference"
      @success="handleInferenceSuccess"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, h, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import { NButton, NTag, NSpace, NIcon, useMessage, useDialog } from 'naive-ui';
import {
  LayersOutline,
  PlayCircleOutline,
  CheckmarkCircleOutline,
  TrendingUpOutline,
  HardwareChipOutline,
  SchoolOutline,
  PersonOutline,
  FlashOutline,
  InformationCircleOutline,
  AddCircleOutline,
  RefreshOutline,
  SaveOutline,
  SearchOutline,
  PlayOutline,
  PauseOutline,
  StopOutline,
  EyeOutline,
  TrashOutline,
  DownloadOutline,
  CloudUploadOutline
} from '@vicons/ionicons5';
import * as echarts from 'echarts';
import {
  createDistillationTask,
  fetchDistillationTasks,
  startDistillationTask,
  stopDistillationTask,
  deleteDistillationTask,
  getAllInferenceTasks,
  deleteInferenceTask
} from '@/service/api/model-distillation';
import InferenceDialog from './components/InferenceDialog.vue';

// ==================== 响应式数据 ====================
const message = useMessage();
const dialog = useDialog();
const router = useRouter();

const activeTab = ref('model-config');

// 统计数据
const taskStats = ref({
  total: 0,
  running: 0,
  completed: 0,
  avgAccuracy: 0,
  gpuUsage: 0
});

// 教师模型配置
const teacherModel = ref({
  modelId: '',
  modelPath: '',
  paramSize: '',
  quantization: 'none'
});

// 学生模型配置
const studentModel = ref({
  modelId: '',
  paramSize: '',
  initMethod: 'random',
  pretrainPath: ''
});

// LoRA 配置
const loraConfig = ref({
  rank: 16,
  alpha: 32,
  dropout: 0.05,
  targetModules: ['q_proj', 'v_proj'],
  layers: 'all',
  biasTrain: 'none'
});

// 知识蒸馏配置
const distillConfig = ref({
  temperature: 4.0,
  hardLabelWeight: 0.5,
  softLabelWeight: 0.5,
  lossType: 'kl_div',
  intermediateLayers: false,
  attentionDistill: false
});

// 训练任务
const tasks = ref<any[]>([]);
const tasksLoading = ref(false);
const taskSearchKeyword = ref('');
const selectedTask = ref<any>(null);

// 创建任务对话框
const showCreateTaskModal = ref(false);
const creatingTask = ref(false);
const taskFormRef = ref<any>(null);
const taskForm = ref({
  taskName: '',
  description: '',
  datasetId: '',
  valDatasetId: '',
  epochs: 10,
  batchSize: 16,
  learningRate: 0.0001,
  weightDecay: 0.01,
  gradAccumSteps: 4,
  maxGradNorm: 1.0,
  lrScheduler: 'cosine',
  optimizer: 'adamw',
  gpuDevices: [],
  autoSaveCheckpoint: true,
  checkpointInterval: 5,
  // Qwen2.5-VL多模型配置
  studentModelType: '',
  studentModelSize: '',
  taskType: 'classification',
  numClasses: 10,
  imageSize: 224,
  distillationType: 'feature',
  featureLossType: 'mse',
  alignFeature: true
});

const taskFormRules = {
  taskName: { required: true, message: '请输入任务名称', trigger: 'blur' },
  datasetId: { required: true, message: '请选择训练数据集', trigger: 'change' },
  epochs: { required: true, type: 'number', message: '请输入训练轮数', trigger: 'blur' }
};

// LoRA 预设
const loraPresets = ref<any[]>([]);
const loraPresetsLoading = ref(false);
const showSaveLoraPresetModal = ref(false);
const savingLoraPreset = ref(false);
const loraPresetName = ref('');
const loraPresetDesc = ref('');

// 图表引用
const lossChartRef = ref<HTMLElement | null>(null);
const accuracyChartRef = ref<HTMLElement | null>(null);
const gpuChartRef = ref<HTMLElement | null>(null);
const memoryChartRef = ref<HTMLElement | null>(null);

let lossChart: echarts.ECharts | null = null;
let accuracyChart: echarts.ECharts | null = null;
let gpuChart: echarts.ECharts | null = null;
let memoryChart: echarts.ECharts | null = null;

// ==================== 已训练模型相关数据 ====================

// 已训练模型搜索条件
const trainedModelsSearch = ref({
  taskName: '',
  teacherModel: null as string | null,
  studentModel: null as string | null,
  minAccuracy: null as number | null
});

// 已训练模型分页
const trainedModelsPagination = {
  pageSize: 10
};

// ==================== 选项数据 ====================

// 教师模型选项
const teacherModelOptions = [
  { label: 'LLaMA-2-7B', value: 'llama2-7b', paramSize: '7B' },
  { label: 'LLaMA-2-13B', value: 'llama2-13b', paramSize: '13B' },
  { label: 'LLaMA-2-70B', value: 'llama2-70b', paramSize: '70B' },
  { label: 'Qwen-7B', value: 'qwen-7b', paramSize: '7B' },
  { label: 'Qwen-14B', value: 'qwen-14b', paramSize: '14B' },
  { label: 'Qwen2.5-VL-8B', value: 'qwen2.5-vl-8b', paramSize: '8B' },
  { label: 'Baichuan2-7B', value: 'baichuan2-7b', paramSize: '7B' },
  { label: 'Baichuan2-13B', value: 'baichuan2-13b', paramSize: '13B' },
  { label: 'ChatGLM3-6B', value: 'chatglm3-6b', paramSize: '6B' },
  { label: 'InternLM-7B', value: 'internlm-7b', paramSize: '7B' }
];

// 学生模型选项
const studentModelOptions = [
  { label: 'TinyLLaMA-1.1B', value: 'tinyllama-1.1b', paramSize: '1.1B' },
  { label: 'MiniGPT-350M', value: 'minigpt-350m', paramSize: '350M' },
  { label: 'DistilBERT-110M', value: 'distilbert-110m', paramSize: '110M' },
  { label: 'BERT-Base-110M', value: 'bert-base-110m', paramSize: '110M' },
  { label: 'GPT-2-Small-117M', value: 'gpt2-small-117m', paramSize: '117M' },
  { label: 'T5-Small-60M', value: 't5-small-60m', paramSize: '60M' },
  { label: 'ResNet (图像分类)', value: 'resnet', paramSize: 'Variable' },
  { label: 'Vision Transformer (图像分类)', value: 'vit', paramSize: 'Variable' },
  { label: 'YOLOv8 (目标检测)', value: 'yolov8', paramSize: 'Variable' },
  { label: 'UNet (图像分割)', value: 'unet', paramSize: 'Variable' },
  { label: 'LSTM (序列特征提取)', value: 'lstm', paramSize: 'Variable' },
  { label: 'Custom Model', value: 'custom', paramSize: 'Custom' }
];

// 量化选项
const quantizationOptions = [
  { label: '无量化 (FP32)', value: 'none' },
  { label: 'FP16 半精度', value: 'fp16' },
  { label: 'INT8 量化', value: 'int8' },
  { label: '4-bit 量化 (QLoRA)', value: '4bit' },
  { label: '8-bit 量化', value: '8bit' }
];

// 初始化方式
const initMethodOptions = [
  { label: '随机初始化', value: 'random' },
  { label: '从教师模型蒸馏', value: 'distill' },
  { label: '加载预训练权重', value: 'pretrained' }
];

// LoRA 目标模块
const loraTargetOptions = [
  { label: 'Query Projection (q_proj)', value: 'q_proj' },
  { label: 'Key Projection (k_proj)', value: 'k_proj' },
  { label: 'Value Projection (v_proj)', value: 'v_proj' },
  { label: 'Output Projection (o_proj)', value: 'o_proj' },
  { label: 'Gate Projection (gate_proj)', value: 'gate_proj' },
  { label: 'Up Projection (up_proj)', value: 'up_proj' },
  { label: 'Down Projection (down_proj)', value: 'down_proj' }
];

// Bias 训练选项
const biasTrainOptions = [
  { label: '不训练 Bias', value: 'none' },
  { label: '仅训练 LoRA Bias', value: 'lora_only' },
  { label: '训练所有 Bias', value: 'all' }
];

// 蒸馏损失类型
const distillLossOptions = [
  { label: 'KL 散度', value: 'kl_div' },
  { label: '均方误差 (MSE)', value: 'mse' },
  { label: '余弦相似度', value: 'cosine' },
  { label: '交叉熵', value: 'cross_entropy' }
];

// 数据集选项（示例）
const datasetOptions = [
  { label: '中文问答数据集 v1', value: 'qa-zh-v1' },
  { label: '中文对话数据集 v2', value: 'dialogue-zh-v2' },
  { label: '英文指令数据集', value: 'instruction-en' },
  { label: '多轮对话数据集', value: 'multi-turn' },
  { label: '自定义数据集', value: 'custom' }
];

// 学习率调度器
const lrSchedulerOptions = [
  { label: 'Cosine 退火', value: 'cosine' },
  { label: 'Linear 衰减', value: 'linear' },
  { label: 'Constant 恒定', value: 'constant' },
  { label: 'Polynomial 多项式', value: 'polynomial' },
  { label: 'Step 阶梯式', value: 'step' }
];

// 优化器选项
const optimizerOptions = [
  { label: 'AdamW', value: 'adamw' },
  { label: 'Adam', value: 'adam' },
  { label: 'SGD', value: 'sgd' },
  { label: 'AdaGrad', value: 'adagrad' },
  { label: 'RMSprop', value: 'rmsprop' }
];

// GPU 选项
const gpuOptions = [
  { label: 'GPU 0', value: '0' },
  { label: 'GPU 1', value: '1' },
  { label: 'GPU 2', value: '2' },
  { label: 'GPU 3', value: '3' }
];

// Qwen2.5-VL学生模型大小选项
const studentModelSizeOptions = ref([
  { label: 'ResNet18', value: 'resnet18' },
  { label: 'ResNet34', value: 'resnet34' },
  { label: 'ResNet50', value: 'resnet50' }
]);

// 任务类型选项
const taskTypeOptions = [
  { label: '图像分类 (Classification)', value: 'classification' },
  { label: '目标检测 (Detection)', value: 'detection' },
  { label: '图像分割 (Segmentation)', value: 'segmentation' }
];

// 蒸馏类型选项
const distillationTypeOptions = [
  { label: '特征蒸馏 (Feature)', value: 'feature' },
  { label: 'Logit蒸馏', value: 'logit' },
  { label: '混合蒸馏 (Hybrid)', value: 'hybrid' }
];

// 特征损失类型选项
const featureLossTypeOptions = [
  { label: '均方误差 (MSE)', value: 'mse' },
  { label: '余弦相似度 (Cosine)', value: 'cosine' }
];

// ==================== 计算属性 ====================

// 检测是否选择了Qwen教师模型
const isQwenTeacher = computed(() => {
  return teacherModel.value.modelId &&
         (teacherModel.value.modelId.toLowerCase().includes('qwen') ||
          teacherModel.value.modelId.toLowerCase().includes('qwen2'));
});

const filteredTasks = computed(() => {
  if (!taskSearchKeyword.value) return tasks.value;
  const keyword = taskSearchKeyword.value.toLowerCase();
  return tasks.value.filter(
    task =>
      task.taskName.toLowerCase().includes(keyword) ||
      task.taskId.toLowerCase().includes(keyword)
  );
});

const taskPagination = {
  pageSize: 10
};

// 已训练模型筛选
const filteredCompletedModels = computed(() => {
  let filtered = tasks.value.filter(task => task.status === 'COMPLETED' && task.accuracy && task.accuracy > 0);

  const search = trainedModelsSearch.value;

  // 按任务名称筛选
  if (search.taskName) {
    filtered = filtered.filter(task =>
      task.taskName.toLowerCase().includes(search.taskName.toLowerCase())
    );
  }

  // 按教师模型筛选
  if (search.teacherModel) {
    filtered = filtered.filter(task => task.teacherModel === search.teacherModel);
  }

  // 按学生模型筛选
  if (search.studentModel) {
    filtered = filtered.filter(task => task.studentModel === search.studentModel);
  }

  // 按最小准确率筛选
  if (search.minAccuracy !== null && search.minAccuracy !== undefined) {
    filtered = filtered.filter(task => task.accuracy >= search.minAccuracy!);
  }

  return filtered;
});

// ==================== 表格列定义 ====================

// 已训练模型表格列
const trainedModelsColumns = [
  {
    title: '任务ID',
    key: 'taskId',
    width: 120,
    ellipsis: { tooltip: true }
  },
  {
    title: '任务名称',
    key: 'taskName',
    width: 180,
    ellipsis: { tooltip: true }
  },
  {
    title: '教师模型',
    key: 'teacherModel',
    width: 150,
    render: (row: any) => {
      const model = teacherModelOptions.find(m => m.value === row.teacherModel);
      return h('span', model?.label || row.teacherModel);
    }
  },
  {
    title: '学生模型',
    key: 'studentModel',
    width: 150,
    render: (row: any) => {
      const model = studentModelOptions.find(m => m.value === row.studentModel);
      return h('span', model?.label || row.studentModel);
    }
  },
  {
    title: '训练轮数',
    key: 'totalEpochs',
    width: 100,
    align: 'center'
  },
  {
    title: 'LoRA Rank',
    key: 'loraRank',
    width: 100,
    align: 'center'
  },
  {
    title: '准确率',
    key: 'accuracy',
    width: 120,
    align: 'center',
    render: (row: any) => {
      const accuracy = row.accuracy || 0;
      const type = accuracy >= 90 ? 'success' : accuracy >= 80 ? 'info' : accuracy >= 70 ? 'warning' : 'error';
      return h(
        NTag,
        { type, size: 'small' },
        { default: () => `${accuracy.toFixed(2)}%` }
      );
    }
  },
  {
    title: '创建时间',
    key: 'createTime',
    width: 180
  },
  {
    title: '操作',
    key: 'actions',
    width: 200,
    align: 'center',
    fixed: 'right',
    render: (row: any) => {
      return h(
        'div',
        { class: 'flex gap-8px justify-center' },
        [
          h(
            NButton,
            {
              size: 'small',
              type: 'primary',
              onClick: () => handleViewTrainedModelDetail(row)
            },
            { default: () => '查看详情' }
          ),
          h(
            NButton,
            {
              size: 'small',
              type: 'info',
              onClick: () => handleUseModelForAnnotation(row)
            },
            { default: () => '用于标注' }
          )
        ]
      );
    }
  }
];

const taskColumns = [
  {
    title: '任务ID',
    key: 'taskId',
    width: 150,
    ellipsis: { tooltip: true }
  },
  {
    title: '任务名称',
    key: 'taskName',
    width: 200
  },
  {
    title: '教师模型',
    key: 'teacherModel',
    width: 150
  },
  {
    title: '学生模型',
    key: 'studentModel',
    width: 150
  },
  {
    title: 'LoRA Rank',
    key: 'loraRank',
    width: 100
  },
  {
    title: '状态',
    key: 'status',
    width: 120,
    render(row: any) {
      return h(
        NTag,
        { type: getStatusType(row.status) },
        { default: () => row.status }
      );
    }
  },
  {
    title: '进度',
    key: 'progress',
    width: 120,
    render(row: any) {
      return `${row.currentEpoch}/${row.totalEpochs} (${row.progress}%)`;
    }
  },
  {
    title: '准确率',
    key: 'accuracy',
    width: 100,
    render(row: any) {
      return row.accuracy ? `${row.accuracy}%` : '-';
    }
  },
  {
    title: '创建时间',
    key: 'createTime',
    width: 180
  },
  {
    title: '操作',
    key: 'actions',
    width: 250,
    fixed: 'right',
    render(row: any) {
      return h(
        NSpace,
        {},
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                disabled: row.status === 'RUNNING',
                onClick: () => handleStartTask(row)
              },
              { default: () => '启动', icon: () => h(NIcon, { component: PlayOutline }) }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'warning',
                disabled: row.status !== 'RUNNING',
                onClick: () => handlePauseTask(row)
              },
              { default: () => '暂停', icon: () => h(NIcon, { component: PauseOutline }) }
            ),
            h(
              NButton,
              {
                size: 'small',
                onClick: () => handleViewTask(row)
              },
              { default: () => '监控', icon: () => h(NIcon, { component: EyeOutline }) }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'error',
                onClick: () => handleDeleteTask(row)
              },
              { default: () => '删除', icon: () => h(NIcon, { component: TrashOutline }) }
            )
          ]
        }
      );
    }
  }
];

const loraPresetColumns = [
  {
    title: '预设名称',
    key: 'presetName',
    width: 200
  },
  {
    title: 'Rank',
    key: 'rank',
    width: 80
  },
  {
    title: 'Alpha',
    key: 'alpha',
    width: 80
  },
  {
    title: 'Dropout',
    key: 'dropout',
    width: 100
  },
  {
    title: '目标模块',
    key: 'targetModules',
    width: 300,
    render(row: any) {
      return row.targetModules.join(', ');
    }
  },
  {
    title: '描述',
    key: 'description',
    ellipsis: { tooltip: true }
  },
  {
    title: '创建时间',
    key: 'createTime',
    width: 180
  },
  {
    title: '操作',
    key: 'actions',
    width: 180,
    fixed: 'right',
    render(row: any) {
      return h(
        NSpace,
        {},
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                onClick: () => handleLoadLoraPreset(row)
              },
              { default: () => '加载配置' }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'error',
                onClick: () => handleDeleteLoraPreset(row)
              },
              { default: () => '删除', icon: () => h(NIcon, { component: TrashOutline }) }
            )
          ]
        }
      );
    }
  }
];

// 推理任务列表列定义
const inferenceColumns = [
  {
    title: '推理ID',
    key: 'inferenceId',
    width: 180,
    ellipsis: { tooltip: true }
  },
  {
    title: '训练任务ID',
    key: 'taskId',
    width: 180,
    ellipsis: { tooltip: true }
  },
  {
    title: '模型类型',
    key: 'modelType',
    width: 120,
    render: (row: any) => {
      const typeMap: Record<string, string> = {
        resnet: 'ResNet',
        vit: 'ViT',
        yolov8: 'YOLOv8',
        unet: 'UNet',
        lstm: 'LSTM'
      };
      return h('span', typeMap[row.modelType] || row.modelType);
    }
  },
  {
    title: '状态',
    key: 'status',
    width: 120,
    render(row: any) {
      const statusMap: Record<string, { type: string; text: string }> = {
        RUNNING: { type: 'info', text: '运行中' },
        COMPLETED: { type: 'success', text: '已完成' },
        FAILED: { type: 'error', text: '失败' }
      };
      const status = statusMap[row.status] || { type: 'default', text: row.status };
      return h(NTag, { type: status.type as any }, { default: () => status.text });
    }
  },
  {
    title: '处理图像数',
    key: 'processedImages',
    width: 120,
    align: 'center'
  },
  {
    title: '成功/失败',
    key: 'result',
    width: 120,
    align: 'center',
    render: (row: any) => {
      return h('span', `${row.successCount || 0}/${row.failureCount || 0}`);
    }
  },
  {
    title: '输出目录',
    key: 'outputDir',
    width: 250,
    ellipsis: { tooltip: true }
  },
  {
    title: '耗时(秒)',
    key: 'duration',
    width: 100,
    align: 'center'
  },
  {
    title: '开始时间',
    key: 'startTime',
    width: 180
  },
  {
    title: '操作',
    key: 'actions',
    width: 150,
    fixed: 'right',
    render(row: any) {
      return h(
        NSpace,
        {},
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                disabled: row.status === 'RUNNING',
                onClick: () => handleViewInferenceResult(row)
              },
              { default: () => '查看结果' }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'error',
                onClick: () => handleDeleteInference(row)
              },
              { default: () => '删除' }
            )
          ]
        }
      );
    }
  }
];

// ==================== 方法 ====================

// 获取状态类型
function getStatusType(status: string | null | undefined) {
  // 添加空值检查
  if (!status) {
    return 'default';
  }

  const statusMap: Record<string, any> = {
    RUNNING: 'success',
    PENDING: 'warning',
    COMPLETED: 'info',
    FAILED: 'error',
    PAUSED: 'default',
    STOPPED: 'warning'
  };
  return statusMap[status] || 'default';
}

// 教师模型变更
function handleTeacherModelChange(value: string) {
  const model = teacherModelOptions.find(m => m.value === value);
  if (model) {
    teacherModel.value.paramSize = model.paramSize;
  }
}

// 学生模型变更
function handleStudentModelChange(value: string) {
  const model = studentModelOptions.find(m => m.value === value);
  if (model) {
    studentModel.value.paramSize = model.paramSize;
  }
  // 如果是Qwen模型，更新studentModelType
  if (isQwenTeacher.value && ['resnet', 'vit', 'yolov8', 'unet', 'lstm'].includes(value)) {
    taskForm.value.studentModelType = value;
    handleStudentModelTypeChange(value);
  }
}

// Qwen学生模型类型变更，更新模型大小选项
function handleStudentModelTypeChange(type: string) {
  const sizeOptionsMap: Record<string, any[]> = {
    resnet: [
      { label: 'ResNet18', value: 'resnet18' },
      { label: 'ResNet34', value: 'resnet34' },
      { label: 'ResNet50', value: 'resnet50' },
      { label: 'ResNet101', value: 'resnet101' }
    ],
    vit: [
      { label: 'ViT-Tiny', value: 'vit-tiny' },
      { label: 'ViT-Small', value: 'vit-small' },
      { label: 'ViT-Base', value: 'vit-base' }
    ],
    yolov8: [
      { label: 'YOLOv8n (Nano)', value: 'n' },
      { label: 'YOLOv8s (Small)', value: 's' },
      { label: 'YOLOv8m (Medium)', value: 'm' },
      { label: 'YOLOv8l (Large)', value: 'l' }
    ],
    unet: [
      { label: 'UNet-Small', value: 'small' },
      { label: 'UNet-Medium', value: 'medium' },
      { label: 'UNet-Large', value: 'large' }
    ],
    lstm: [
      { label: 'LSTM-Small (256)', value: 'small' },
      { label: 'LSTM-Medium (512)', value: 'medium' },
      { label: 'LSTM-Large (1024)', value: 'large' }
    ]
  };

  studentModelSizeOptions.value = sizeOptionsMap[type] || [];
  if (studentModelSizeOptions.value.length > 0) {
    taskForm.value.studentModelSize = studentModelSizeOptions.value[0].value;
  }
}

// 重置配置
function handleResetConfig() {
  dialog.warning({
    title: '确认重置',
    content: '确定要重置所有配置到默认值吗？',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: () => {
      // 重置所有配置
      teacherModel.value = {
        modelId: '',
        modelPath: '',
        paramSize: '',
        quantization: 'none'
      };
      studentModel.value = {
        modelId: '',
        paramSize: '',
        initMethod: 'random',
        pretrainPath: ''
      };
      loraConfig.value = {
        rank: 16,
        alpha: 32,
        dropout: 0.05,
        targetModules: ['q_proj', 'v_proj'],
        layers: 'all',
        biasTrain: 'none'
      };
      distillConfig.value = {
        temperature: 4.0,
        hardLabelWeight: 0.5,
        softLabelWeight: 0.5,
        lossType: 'kl_div',
        intermediateLayers: false,
        attentionDistill: false
      };
      message.success('配置已重置');
    }
  });
}

// 保存配置
function handleSaveConfig() {
  // TODO: 调用后端API保存配置
  message.success('配置保存成功');
}

/**
 * 安全地提取错误消息字符串
 * @param error - 可能是字符串、Error对象或其他类型
 * @returns 字符串形式的错误消息
 */
function getErrorMessage(error: any): string {
  if (!error) return '';
  if (typeof error === 'string') return error;
  if (error.message) return String(error.message);
  try {
    return String(error);
  } catch {
    return '未知错误';
  }
}

// 刷新任务列表
async function refreshTasks() {
  tasksLoading.value = true;
  try {
    const res = await fetchDistillationTasks();
    console.log('获取训练任务列表响应:', res);

    // 兼容不同的响应格式
    if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
      tasks.value = res.data || [];
      console.log('训练任务列表:', tasks.value);

      // 如果当前有选中的任务，需要同步更新 selectedTask
      if (selectedTask.value && selectedTask.value.taskId) {
        const updatedTask = tasks.value.find(t => t.taskId === selectedTask.value.taskId);
        if (updatedTask) {
          // 更新为最新数据
          selectedTask.value = updatedTask;
        } else {
          // 任务已被删除，清空选中
          selectedTask.value = null;
        }
      }
    } else {
      message.error(res.message || getErrorMessage(res.error) || '获取任务列表失败');
      tasks.value = [];
    }
  } catch (error: any) {
    console.error('获取任务列表错误:', error);
    message.error('获取任务列表失败：' + (error?.message || '未知错误'));
    tasks.value = [];
  } finally {
    tasksLoading.value = false;
  }
}

// 创建任务
async function handleCreateTask() {
  try {
    // 表单验证
    await taskFormRef.value?.validate();

    creatingTask.value = true;

    // 准备提交数据
    const submitData: any = {
      ...taskForm.value,
      // 从Tab1的模型配置中获取
      teacherModel: teacherModel.value.modelId,
      studentModel: studentModel.value.modelId,
      teacherParamSize: teacherModel.value.paramSize,
      teacherModelPath: teacherModel.value.modelPath,
      teacherQuantization: teacherModel.value.quantization,
      studentParamSize: studentModel.value.paramSize,
      studentInitMethod: studentModel.value.initMethod,
      studentPretrainPath: studentModel.value.pretrainPath,
      // LoRA配置
      loraRank: loraConfig.value.rank,
      loraAlpha: loraConfig.value.alpha,
      loraDropout: loraConfig.value.dropout,
      loraTargetModules: loraConfig.value.targetModules?.join(','),
      loraLayers: loraConfig.value.layers,
      loraBiasTrain: loraConfig.value.biasTrain,
      // 知识蒸馏配置
      temperature: distillConfig.value.temperature,
      alpha: distillConfig.value.softLabelWeight,
      hardLabelWeight: distillConfig.value.hardLabelWeight,
      softLabelWeight: distillConfig.value.softLabelWeight,
      distillLossType: distillConfig.value.lossType,
      intermediateLayers: distillConfig.value.intermediateLayers,
      attentionDistill: distillConfig.value.attentionDistill,
      // GPU设备转为逗号分隔字符串
      gpuDevices: taskForm.value.gpuDevices?.join(',')
    };

    console.log('准备提交训练任务数据:', submitData);

    // 调用后端API创建训练任务
    const res = await createDistillationTask(submitData);

    console.log('创建训练任务响应:', res);

    // 兼容不同的响应格式
    if (res.code === 200 || res.code === 0 || (res.data && !res.error)) {
      message.success('训练任务创建成功！');
      showCreateTaskModal.value = false;
      // 使用 nextTick 确保模态框关闭后再刷新列表
      await nextTick();
      await refreshTasks();
    } else {
      message.error(res.message || getErrorMessage(res.error) || '创建任务失败');
    }
  } catch (error: any) {
    console.error('创建任务失败:', error);
    message.error('创建任务失败：' + (error?.message || '未知错误'));
  } finally {
    creatingTask.value = false;
  }
}

// 启动任务
async function handleStartTask(task: any) {
  try {
    console.log('启动训练任务:', task.taskId);
    const res = await startDistillationTask(task.taskId);

    // 兼容不同的响应格式
    if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
      message.success(`任务 "${task.taskName}" 已启动`);
      // 使用 setTimeout 延迟刷新，避免与按钮点击事件的 DOM 操作冲突
      setTimeout(() => {
        refreshTasks();
      }, 100);
    } else {
      message.error(res.message || getErrorMessage(res.error) || '启动任务失败');
    }
  } catch (error: any) {
    console.error('启动任务失败:', error);
    message.error('启动任务失败：' + (error?.message || '未知错误'));
  }
}

// 停止任务
async function handleStopTask(task: any) {
  dialog.warning({
    title: '确认停止',
    content: `确定要停止任务 "${task.taskName}" 吗？`,
    positiveText: '停止',
    negativeText: '取消',
    onPositiveClick: () => {
      // 返回 Promise，让 Dialog 等待异步操作完成后再关闭
      return new Promise<void>(async (resolve, reject) => {
        try {
          console.log('停止训练任务:', task.taskId);
          const res = await stopDistillationTask(task.taskId);

          // 兼容不同的响应格式
          if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
            message.success(`任务 "${task.taskName}" 已停止`);
            // 等待 Dialog 完全关闭后再刷新列表
            resolve();
            // 使用 setTimeout 确保 Dialog 完全关闭后再刷新
            setTimeout(() => {
              refreshTasks();
            }, 100);
          } else {
            message.error(res.message || getErrorMessage(res.error) || '停止任务失败');
            resolve();
          }
        } catch (error: any) {
          console.error('停止任务失败:', error);
          message.error('停止任务失败：' + (error?.message || '未知错误'));
          reject(error);
        }
      });
    }
  });
}

// 暂停任务
async function handlePauseTask(task: any) {
  dialog.warning({
    title: '确认暂停',
    content: `确定要暂停任务 "${task.taskName}" 吗？可以稍后恢复运行。`,
    positiveText: '暂停',
    negativeText: '取消',
    onPositiveClick: () => {
      // 返回 Promise，让 Dialog 等待异步操作完成后再关闭
      return new Promise<void>(async (resolve, reject) => {
        try {
          console.log('暂停训练任务:', task.taskId);
          // 暂停也使用 stop 接口，因为后端可能没有单独的 pause 接口
          const res = await stopDistillationTask(task.taskId);

          // 兼容不同的响应格式
          if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
            message.success(`任务 "${task.taskName}" 已暂停`);
            resolve();
            // 使用 setTimeout 确保 Dialog 完全关闭后再刷新
            setTimeout(() => {
              refreshTasks();
            }, 100);
          } else {
            message.error(res.message || getErrorMessage(res.error) || '暂停任务失败');
            resolve();
          }
        } catch (error: any) {
          console.error('暂停任务失败:', error);
          message.error('暂停任务失败：' + (error?.message || '未知错误'));
          reject(error);
        }
      });
    }
  });
}

// 查看任务
function handleViewTask(task: any) {
  selectedTask.value = task;
  activeTab.value = 'training-monitor';
  // 使用 nextTick 确保 DOM 渲染完成后再初始化图表
  nextTick(() => {
    initCharts();
  });
}

// 删除任务
async function handleDeleteTask(task: any) {
  dialog.warning({
    title: '确认删除',
    content: `确定要删除任务 "${task.taskName}" 吗？此操作不可恢复！`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: () => {
      // 返回 Promise，让 Dialog 等待异步操作完成后再关闭
      return new Promise<void>(async (resolve, reject) => {
        try {
          console.log('删除训练任务:', task.taskId);
          const res = await deleteDistillationTask(task.taskId);

          // 兼容不同的响应格式
          if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
            message.success(`任务 "${task.taskName}" 已删除`);
            resolve();
            // 使用 setTimeout 确保 Dialog 完全关闭后再刷新
            setTimeout(() => {
              refreshTasks();
            }, 100);
          } else {
            message.error(res.message || getErrorMessage(res.error) || '删除任务失败');
            resolve();
          }
        } catch (error: any) {
          console.error('删除任务失败:', error);
          message.error('删除任务失败：' + (error?.message || '未知错误'));
          reject(error);
        }
      });
    }
  });
}

// 刷新 LoRA 预设
function refreshLoraPresets() {
  loraPresetsLoading.value = true;
  // TODO: 调用后端API
  setTimeout(() => {
    loraPresets.value = [
      {
        presetName: '标准配置',
        rank: 16,
        alpha: 32,
        dropout: 0.05,
        targetModules: ['q_proj', 'v_proj'],
        description: '适用于大多数场景的标准 LoRA 配置',
        createTime: '2025-11-18 15:00:00'
      }
    ];
    loraPresetsLoading.value = false;
  }, 500);
}

// 保存 LoRA 预设
function handleSaveLoraPreset() {
  if (!loraPresetName.value) {
    message.error('请输入预设名称');
    return;
  }
  savingLoraPreset.value = true;
  // TODO: 调用后端API
  setTimeout(() => {
    message.success('LoRA 预设保存成功');
    showSaveLoraPresetModal.value = false;
    savingLoraPreset.value = false;
    loraPresetName.value = '';
    loraPresetDesc.value = '';
    refreshLoraPresets();
  }, 1000);
}

// 加载 LoRA 预设
function handleLoadLoraPreset(preset: any) {
  loraConfig.value = {
    rank: preset.rank,
    alpha: preset.alpha,
    dropout: preset.dropout,
    targetModules: preset.targetModules,
    layers: 'all',
    biasTrain: 'none'
  };
  activeTab.value = 'model-config';
  message.success(`已加载预设: ${preset.presetName}`);
}

// 删除 LoRA 预设
function handleDeleteLoraPreset(preset: any) {
  dialog.warning({
    title: '确认删除',
    content: `确定要删除预设 "${preset.presetName}" 吗？`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: () => {
      message.success('预设已删除');
      refreshLoraPresets();
    }
  });
}

// 初始化图表
function initCharts() {
  setTimeout(() => {
    try {
      initLossChart();
      initAccuracyChart();
      initGpuChart();
      initMemoryChart();
    } catch (error) {
      console.error('图表初始化失败:', error);
      // 即使图表初始化失败，也不影响其他功能
    }
  }, 100);
}

// 初始化损失曲线图表
function initLossChart() {
  if (!lossChartRef.value) {
    console.warn('lossChartRef 未找到，跳过初始化');
    return;
  }

  try {
    // 如果已存在实例，先销毁
    if (lossChart) {
      lossChart.dispose();
    }
    lossChart = echarts.init(lossChartRef.value);
  } catch (error) {
    console.error('损失曲线图表初始化失败:', error);
    return;
  }

  const option = {
    title: {
      text: '训练损失曲线',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['总损失', '硬标签损失', '软标签损失 (蒸馏)'],
      bottom: 10
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.from({ length: 100 }, (_, i) => i + 1),
      name: 'Steps'
    },
    yAxis: {
      type: 'value',
      name: 'Loss'
    },
    series: [
      {
        name: '总损失',
        type: 'line',
        data: Array.from({ length: 100 }, () => Math.random() * 2 + 1),
        smooth: true,
        lineStyle: { width: 2 }
      },
      {
        name: '硬标签损失',
        type: 'line',
        data: Array.from({ length: 100 }, () => Math.random() * 1.5 + 0.5),
        smooth: true,
        lineStyle: { width: 2 }
      },
      {
        name: '软标签损失 (蒸馏)',
        type: 'line',
        data: Array.from({ length: 100 }, () => Math.random() * 1.2 + 0.3),
        smooth: true,
        lineStyle: { width: 2 }
      }
    ]
  };

  lossChart.setOption(option);
}

// 初始化准确率图表
function initAccuracyChart() {
  if (!accuracyChartRef.value) {
    console.warn('accuracyChartRef 未找到，跳过初始化');
    return;
  }

  try {
    if (accuracyChart) {
      accuracyChart.dispose();
    }
    accuracyChart = echarts.init(accuracyChartRef.value);
  } catch (error) {
    console.error('准确率曲线图表初始化失败:', error);
    return;
  }

  const option = {
    title: {
      text: '教师-学生模型准确率对比',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['教师模型', '学生模型'],
      bottom: 10
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.from({ length: 10 }, (_, i) => `Epoch ${i + 1}`),
      name: 'Epoch'
    },
    yAxis: {
      type: 'value',
      name: 'Accuracy (%)',
      min: 0,
      max: 100
    },
    series: [
      {
        name: '教师模型',
        type: 'line',
        data: [65, 72, 78, 82, 85, 87, 89, 90, 91, 92],
        smooth: true,
        lineStyle: { width: 3, color: '#409EFF' },
        itemStyle: { color: '#409EFF' }
      },
      {
        name: '学生模型',
        type: 'line',
        data: [45, 58, 65, 71, 76, 80, 83, 85, 87, 88],
        smooth: true,
        lineStyle: { width: 3, color: '#67C23A' },
        itemStyle: { color: '#67C23A' }
      }
    ]
  };

  accuracyChart.setOption(option);
}

// 初始化 GPU 使用率图表
function initGpuChart() {
  if (!gpuChartRef.value) {
    console.warn('gpuChartRef 未找到，跳过初始化');
    return;
  }

  try {
    if (gpuChart) {
      gpuChart.dispose();
    }
    gpuChart = echarts.init(gpuChartRef.value);
  } catch (error) {
    console.error('GPU使用率图表初始化失败:', error);
    return;
  }

  const option = {
    tooltip: {
      formatter: '{b}: {c}%'
    },
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 100,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 6,
            color: [
              [0.3, '#67C23A'],
              [0.7, '#E6A23C'],
              [1, '#F56C6C']
            ]
          }
        },
        pointer: {
          icon: 'path://M2090.36389,615.30999 L2090.36389,615.30999 C2091.48372,615.30999 2092.40383,616.194028 2092.44859,617.312956 L2096.90698,728.755929 C2097.05155,732.369577 2094.2393,735.416212 2090.62566,735.56078 C2090.53845,735.564269 2090.45117,735.566014 2090.36389,735.566014 L2090.36389,735.566014 C2086.74736,735.566014 2083.81557,732.63423 2083.81557,729.017692 C2083.81557,728.930412 2083.81732,728.84314 2083.82081,728.755929 L2088.2792,617.312956 C2088.32396,616.194028 2089.24407,615.30999 2090.36389,615.30999 Z',
          length: '75%',
          width: 16,
          offsetCenter: [0, '5%']
        },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          color: 'auto',
          fontSize: 24,
          offsetCenter: [0, '80%']
        },
        data: [{ value: 78, name: 'GPU 使用率' }]
      }
    ]
  };

  gpuChart.setOption(option);
}

// 初始化显存使用图表
function initMemoryChart() {
  if (!memoryChartRef.value) {
    console.warn('memoryChartRef 未找到，跳过初始化');
    return;
  }

  try {
    if (memoryChart) {
      memoryChart.dispose();
    }
    memoryChart = echarts.init(memoryChartRef.value);
  } catch (error) {
    console.error('内存使用率图表初始化失败:', error);
    return;
  }

  const option = {
    tooltip: {
      formatter: '{b}: {c} GB'
    },
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 24,
        splitNumber: 8,
        axisLine: {
          lineStyle: {
            width: 6,
            color: [
              [0.5, '#67C23A'],
              [0.8, '#E6A23C'],
              [1, '#F56C6C']
            ]
          }
        },
        pointer: {
          icon: 'path://M2090.36389,615.30999 L2090.36389,615.30999 C2091.48372,615.30999 2092.40383,616.194028 2092.44859,617.312956 L2096.90698,728.755929 C2097.05155,732.369577 2094.2393,735.416212 2090.62566,735.56078 C2090.53845,735.564269 2090.45117,735.566014 2090.36389,735.566014 L2090.36389,735.566014 C2086.74736,735.566014 2083.81557,732.63423 2083.81557,729.017692 C2083.81557,728.930412 2083.81732,728.84314 2083.82081,728.755929 L2088.2792,617.312956 C2088.32396,616.194028 2089.24407,615.30999 2090.36389,615.30999 Z',
          length: '75%',
          width: 16,
          offsetCenter: [0, '5%']
        },
        detail: {
          valueAnimation: true,
          formatter: '{value} GB',
          color: 'auto',
          fontSize: 24,
          offsetCenter: [0, '80%']
        },
        data: [{ value: 18.5, name: '显存使用' }]
      }
    ]
  };

  memoryChart.setOption(option);
}

// ==================== 已训练模型相关方法 ====================

// 重置搜索条件
function resetTrainedModelsSearch() {
  trainedModelsSearch.value = {
    taskName: '',
    teacherModel: null,
    studentModel: null,
    minAccuracy: null
  };
  message.success('已重置搜索条件');
}

// 查看训练模型详情
function handleViewTrainedModelDetail(row: any) {
  dialog.info({
    title: `模型详情 - ${row.taskName}`,
    content: () =>
      h('div', { class: 'space-y-4' }, [
        h('div', { class: 'grid grid-cols-2 gap-4' }, [
          h('div', [
            h('strong', '任务ID: '),
            h('span', row.taskId)
          ]),
          h('div', [
            h('strong', '任务名称: '),
            h('span', row.taskName)
          ]),
          h('div', [
            h('strong', '教师模型: '),
            h('span', teacherModelOptions.find(m => m.value === row.teacherModel)?.label || row.teacherModel)
          ]),
          h('div', [
            h('strong', '学生模型: '),
            h('span', studentModelOptions.find(m => m.value === row.studentModel)?.label || row.studentModel)
          ]),
          h('div', [
            h('strong', '训练轮数: '),
            h('span', row.totalEpochs)
          ]),
          h('div', [
            h('strong', 'LoRA Rank: '),
            h('span', row.loraRank)
          ]),
          h('div', [
            h('strong', '准确率: '),
            h('span', `${row.accuracy.toFixed(2)}%`)
          ]),
          h('div', [
            h('strong', '创建时间: '),
            h('span', row.createTime)
          ])
        ])
      ]),
    style: { width: '600px' }
  });
}

// 使用模型进行标注 - 推理功能
const showInferenceDialog = ref(false);
const selectedTaskForInference = ref<any>(null);

function handleUseModelForAnnotation(row: any) {
  // 打开推理对话框
  selectedTaskForInference.value = row;
  showInferenceDialog.value = true;
}

function handleInferenceSuccess(inferenceId: string) {
  message.success(`推理任务已提交！推理ID: ${inferenceId}`);
  // 切换到推理任务列表Tab
  activeTab.value = 'inference-tasks';
  // 刷新推理任务列表
  refreshInferenceTasks();
}

// 推理任务列表数据
const inferenceTasksData = ref([]);
const inferenceTasksLoading = ref(false);

// 刷新推理任务列表
async function refreshInferenceTasks() {
  inferenceTasksLoading.value = true;
  try {
    const res = await getAllInferenceTasks();
    console.log('获取推理任务列表响应:', res);

    // 兼容不同的响应格式
    if (res.code === 200 || res.code === 0 || (res.data !== undefined && !res.error)) {
      inferenceTasksData.value = res.data || [];
      console.log('推理任务列表:', inferenceTasksData.value);
    } else {
      message.error(res.message || getErrorMessage(res.error) || '获取推理任务列表失败');
    }
  } catch (error: any) {
    console.error('获取推理任务列表错误:', error);
    message.error('获取推理任务列表失败：' + (error?.message || '未知错误'));
  } finally {
    inferenceTasksLoading.value = false;
  }
}

// 查看推理结果
function handleViewInferenceResult(row: any) {
  if (row.status === 'COMPLETED') {
    message.info(`输出目录: ${row.outputDir}\n处理图像: ${row.processedImages} 张\n成功: ${row.successCount} 失败: ${row.failureCount}`);
  } else if (row.status === 'FAILED') {
    message.error(`推理失败: ${row.errorMessage || '未知错误'}`);
  } else {
    message.info('推理任务进行中...');
  }
}

// 删除推理任务
async function handleDeleteInference(row: any) {
  dialog.warning({
    title: '确认删除',
    content: `确定要删除推理任务 ${row.inferenceId} 吗？`,
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        const res = await deleteInferenceTask(row.inferenceId);
        if (res.code === 200 || res.code === 0) {
          message.success('删除成功');
          refreshInferenceTasks();
        } else {
          message.error(res.message || '删除失败');
        }
      } catch (error: any) {
        message.error('删除失败：' + (error?.message || '未知错误'));
      }
    }
  });
}

// ==================== 生命周期 ====================

onMounted(() => {
  refreshTasks();
  refreshLoraPresets();

  // 模拟统计数据
  taskStats.value = {
    total: 15,
    running: 3,
    completed: 10,
    avgAccuracy: 85.6,
    gpuUsage: 78
  };
});

onUnmounted(() => {
  if (lossChart) lossChart.dispose();
  if (accuracyChart) accuracyChart.dispose();
  if (gpuChart) gpuChart.dispose();
  if (memoryChart) memoryChart.dispose();
});
</script>

<style scoped>
.model-distillation-container {
  padding: 16px;
}

.flex {
  display: flex;
}

.items-center {
  align-items: center;
}

.justify-between {
  justify-content: space-between;
}

.mb-2 {
  margin-bottom: 8px;
}

.mb-4 {
  margin-bottom: 16px;
}

.mr-2 {
  margin-right: 8px;
}
</style>
