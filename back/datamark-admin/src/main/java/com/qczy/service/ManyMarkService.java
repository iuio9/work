package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.ManyMarkTaskRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.ManyCreateListResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.model.response.TeamUserResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 16:31
 * @Description:
 */
public interface ManyMarkService {

    /**
     * 判断当前数据集是否在执行任务
     */
    int countBySonIdTask(String sonId);


    /**
     * 判断当前数据集租下的数据级是否在执行任务
     */
    int countBySonIdsTask(String groupId);

    /**
     * 创建多人标注任务
     */
    int addManyMarkTask(ManyMarkTaskRequest request);

    IPage<ManyCreateListResponse> getMyCreateTaskList(Page<ManyCreateListResponse> pageParam);

    /**
     * 终止任务
     */
    int endTask(Integer taskId);

    /**
     * 删除
     */
    int deleteTask(Integer taskId);

    /**
     * 我接收的任务
     */
    IPage<ManyReceiveListResponse> getMyReceiveList(Page<ManyReceiveListResponse> pageParam);

    /**
     * 判断当前用户是否可以转交
     */
    boolean isRelayed(TaskShiftRequest request);

    /**
     * 任务转交
     */
    int taskShift(TaskShiftRequest request);

    /**
     * 根据任务id获取团队人员
     */
    List<TeamUserResponse> getByTaskIdTeamList(Integer taskId,Integer teamType);

    /**
     * 查看任务-结束任务
     */
    int endUserTask(Integer id);

    /**
     * 撤回
     */
    int withdraw(DeleteRequest request);

    /**
     * 每人分配数量
     */
    String allocationNum(String sonId, String teamId);

    /**
     * 判断当前数据集是否符合条件
     */
    boolean isDataSet(String sonId);

}
