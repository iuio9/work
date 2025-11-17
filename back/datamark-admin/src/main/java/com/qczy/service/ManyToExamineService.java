package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ManyFileEntity;
import com.qczy.model.response.DataDetailsResponse;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/12 10:52
 * @Description:
 */
public interface ManyToExamineService {



    /**
     * 验收任务 - 列表
     */
    int isApprove(ManyFileEntity manyFileEntity);

    /**
     * 验收通过
     */
    int verifyComplete(Integer taskId, Integer verifyState);

    /**
     * 打回任务
     */
    int returnTask(Integer taskId, Integer returnState,Integer id);

    /**
     * 剩余验收通过
     */
    int remainingApprove(Integer taskId,Integer id);

    /**
     *  查询当前是否还有未验收的数据
     */
    boolean isRemaining(Integer taskId,Integer id);

    /**
     *  验证是否还有打回的数据
     */
    boolean returnTaskState(Integer taskId, Integer returnState,Integer id);
    /**
     *提交任务
     */
    int submitTask(Integer id);

    String submitTaskPrompt(Integer id);
}
