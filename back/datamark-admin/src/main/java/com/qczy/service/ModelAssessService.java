package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.request.AssessTaskRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.ModelAssessResponse;
import com.qczy.model.response.ModelDebugLogResponse;
import com.qczy.model.response.ModelReportResponse;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 11:21
 * @Description:
 */
public interface ModelAssessService {

    // 创建评估任务
    int createAssessTask(AssessTaskRequest request);

    // 编辑
    int editTask(AssessTaskRequest request);

    // 模型列表
    IPage<ModelAssessResponse> listPage(Page<ModelAssessResponse> pageParam, ModelAssessTaskEntity modelAssessTaskEntity);

    // 删除任务
    int delTask(DeleteRequest request);

    /**
     *  测试评估报告-列表
     */
    IPage<ModelReportResponse> reportListPage(Page<ModelReportResponse> pageParam);

    /**
     * 生成评估报告
     */
    void generateWord(Integer id, HttpServletRequest request, HttpServletResponse response);

    /**
     * 以申请单号生成评估报告
     */
    void generateApplyNoWordZip(Integer id, HttpServletRequest request, HttpServletResponse response);

    /**
     *  查询当前任务状态
     */
    int getTaskStatus(Integer id);


    /**
     * 从调试结果里面获取 请求路径和 请求方式
     */
    ModelDebugLogResponse getModelDebugInfo(AssessTaskRequest request);

    // 判断任务名称是否存在
    boolean isTaskNameRepeat(String taskName, Integer id);

    // 查看任务详情
    AssessTaskRequest taskDetails(Integer id);

    int deleteFile(DeleteRequest request);
}
