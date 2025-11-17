package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ManyAuditEntity;
import com.qczy.model.entity.TeamUserEntity;
import com.qczy.model.request.ExamineReturnRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.ExamineTeamInfoResponse;
import com.qczy.model.response.ManyAuditDetailsResponse;
import com.qczy.model.response.ManyReceiveListResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 15:06
 * @Description:
 */
public interface ManyAllocationService {

    /**
     * 分配审核任务
     */
    int distributionExamine(Integer taskId);

    /**
     * 审核详情
     */
    IPage<ManyAuditDetailsResponse> examineDetails(Page<ManyAuditDetailsResponse> pageParam, Integer taskId);

    /**
     * 我的审核 -列表
     */
    IPage<ManyReceiveListResponse> myExamineTaskList(Page<ManyReceiveListResponse> pageParam);

    /**
     * 我的审核 - 提交任务
     */
    int submitExamineTask(Integer taskId);

    /**
     * 我的审核 - 提交任务
     */
    String submitExamineTaskPrompt(Integer id);

    /**
     * 审核团队基础信息
     */
    ExamineTeamInfoResponse examineTeamInfo(Integer id);

    /**
     * 审核人员分配的列表
     */
    List<ManyAuditEntity> examineTeamList(Integer id);

    /**
     * 确认分配审核
     */
    int confirmAudit(Integer id);


    /**
     * 审核任务转交
     */
    int examineTaskShift(TaskShiftRequest request);


    /**
     *  判断所有任务是否提交了
     */
    boolean submitTaskShift(Integer taskId);

    /**
     *  根据ids 判断当前勾选的审核人员状态
     */
    boolean isExamineStatus(ExamineReturnRequest request);


    /**
     *  退回审核
     */
    int examineReturn(ExamineReturnRequest request);

    /**
     *  判断是否审核人员全部提交了
     */
    boolean isExamineSubmit(Integer taskId);

    /**
     *  审核通过
     */
    int approved(Integer taskId);

}
