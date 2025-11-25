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
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, h } from 'vue';
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

// ==================== 响应式数据 ====================
const message = useMessage();
const dialog = useDialog();

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
  checkpointInterval: 5
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

// ==================== 计算属性 ====================

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

// ==================== 方法 ====================

// 获取状态类型
function getStatusType(status: string) {
  const statusMap: Record<string, any> = {
    RUNNING: 'success',
    PENDING: 'warning',
    COMPLETED: 'info',
    FAILED: 'error',
    PAUSED: 'default'
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

// 刷新任务列表
function refreshTasks() {
  tasksLoading.value = true;
  // TODO: 调用后端API获取任务列表
  setTimeout(() => {
    // 示例数据：包含已完成和运行中的任务（用于演示）
    tasks.value = [
      // 已完成的任务（用于"已训练模型"标签页展示）
      {
        taskId: 'TASK_001',
        taskName: '目标检测协同训练-YOLOv5',
        teacherModel: 'llama2-7b',
        studentModel: 'yolov5s',
        loraRank: 16,
        status: 'COMPLETED',
        progress: 100,
        currentEpoch: 50,
        totalEpochs: 50,
        accuracy: 92.5,
        createTime: '2025-11-23 10:30:00'
      },
      {
        taskId: 'TASK_002',
        taskName: '图像分类协同训练-ResNet',
        teacherModel: 'qwen-7b',
        studentModel: 'resnet50',
        loraRank: 8,
        status: 'COMPLETED',
        progress: 100,
        currentEpoch: 40,
        totalEpochs: 40,
        accuracy: 88.3,
        createTime: '2025-11-20 14:15:00'
      },
      {
        taskId: 'TASK_003',
        taskName: '语义分割协同训练-UNet',
        teacherModel: 'llama2-13b',
        studentModel: 'unet',
        loraRank: 16,
        status: 'COMPLETED',
        progress: 100,
        currentEpoch: 60,
        totalEpochs: 60,
        accuracy: 85.7,
        createTime: '2025-11-18 09:00:00'
      },
      {
        taskId: 'TASK_005',
        taskName: '视觉Transformer协同训练',
        teacherModel: 'llama2-7b',
        studentModel: 'vit',
        loraRank: 16,
        status: 'COMPLETED',
        progress: 100,
        currentEpoch: 45,
        totalEpochs: 45,
        accuracy: 90.2,
        createTime: '2025-11-15 16:45:00'
      },
      // 正在运行的任务
      {
        taskId: 'task-001',
        taskName: 'LLaMA2-7B 蒸馏训练',
        teacherModel: 'llama2-7b',
        studentModel: 'tinyllama',
        loraRank: 16,
        status: 'RUNNING',
        progress: 45,
        currentEpoch: 4,
        totalEpochs: 10,
        accuracy: 82.5,
        createTime: '2025-11-25 10:30:00'
      }
    ];
    tasksLoading.value = false;
  }, 500);
}

// 创建任务
function handleCreateTask() {
  // TODO: 表单验证和API调用
  creatingTask.value = true;
  setTimeout(() => {
    message.success('训练任务创建成功');
    showCreateTaskModal.value = false;
    creatingTask.value = false;
    refreshTasks();
  }, 1000);
}

// 启动任务
function handleStartTask(task: any) {
  message.info(`启动任务: ${task.taskName}`);
  // TODO: 调用后端API
}

// 暂停任务
function handlePauseTask(task: any) {
  message.info(`暂停任务: ${task.taskName}`);
  // TODO: 调用后端API
}

// 查看任务
function handleViewTask(task: any) {
  selectedTask.value = task;
  activeTab.value = 'training-monitor';
  initCharts();
}

// 删除任务
function handleDeleteTask(task: any) {
  dialog.warning({
    title: '确认删除',
    content: `确定要删除任务 "${task.taskName}" 吗？`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: () => {
      message.success('任务已删除');
      refreshTasks();
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
    initLossChart();
    initAccuracyChart();
    initGpuChart();
    initMemoryChart();
  }, 100);
}

// 初始化损失曲线图表
function initLossChart() {
  if (!lossChartRef.value) return;
  lossChart = echarts.init(lossChartRef.value);

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
  if (!accuracyChartRef.value) return;
  accuracyChart = echarts.init(accuracyChartRef.value);

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
  if (!gpuChartRef.value) return;
  gpuChart = echarts.init(gpuChartRef.value);

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
  if (!memoryChartRef.value) return;
  memoryChart = echarts.init(memoryChartRef.value);

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

// 使用模型进行标注
function handleUseModelForAnnotation(row: any) {
  const router = useRouter();
  router.push({
    path: '/data-ano/autoano',
    query: {
      distillationModelId: row.taskId,
      distillationModelName: row.taskName
    }
  });
  message.success(`已选择模型 ${row.taskName}，正在跳转到自动标注页面...`);
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
