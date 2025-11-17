package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.ExamineReturnRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.ExamineTeamInfoResponse;
import com.qczy.model.response.ManyAuditDetailsResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.service.ManyAllocationService;
import com.qczy.utils.CurrentLoginUserUtils;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.beans.Transient;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 15:06
 * @Description:
 */
@Service
public class ManyAllocationServiceImpl implements ManyAllocationService {

    @Autowired
    private ManyMarkMapper manyMarkMapper;
    @Autowired
    private TeamUserMapper teamUserMapper;
    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private ManyAuditMapper manyAuditMapper;
    @Autowired
    private ManyFileMapper manyFileMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;


    @Override
    public IPage<ManyAuditDetailsResponse> examineDetails(Page<ManyAuditDetailsResponse> pageParam, Integer taskId) {
        return manyAuditMapper.examineDetails(pageParam, taskId);
    }

    @Override
    public IPage<ManyReceiveListResponse> myExamineTaskList(Page<ManyReceiveListResponse> pageParam) {
        // 当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        return manyAuditMapper.myExamineTaskList(pageParam, userId);
    }

    @Override
    public int submitExamineTask(Integer id) {
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(auditEntity.getManyMarkId());
        if (ObjectUtils.isEmpty(auditEntity)) {
            return 0;
        }
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        // 修改状态
        auditEntity.setIsSubmit(1);
        auditEntity.setAuditState(4); // TODO 审核结束
        manyAuditMapper.updateById(auditEntity);
      /*  // 如果所有审核人员全部提交，更改任务状态
        Integer count = manyAuditMapper.selectCount(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getManyMarkId, auditEntity.getManyMarkId())
                        .eq(ManyAuditEntity::getIsLose, 0)
                        .eq(ManyAuditEntity::getIsSubmit, 0)
        );
        if (count == 0) {
            // 证明所有审核人员任务都已经提交
            manyMarkEntity.setTaskState(7);
            manyMarkMapper.updateById(manyMarkEntity);
        }*/
        return 1;
    }

    @Override
    public String submitExamineTaskPrompt(Integer id) {
        ManyAuditEntity manyAuditEntity = manyAuditMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyAuditEntity)) {
            return "当前任务不存在, 不可提交！";
        }
        // 获取所有未审核
        int noExamine = manyAuditEntity.getNoExamine();
        if (noExamine == 0) {
            return "提交后不可在进行审核！确认要提交吗？";
        }
        return "当前您有" + noExamine + "项未审核, 提交后不可在进行审核！确认要提交吗？";
    }

    @Override
    public ExamineTeamInfoResponse examineTeamInfo(Integer id) {
        return manyAuditMapper.examineTeamInfo(id);
    }

    @Override
    public List<ManyAuditEntity> examineTeamList(Integer id) {
        return manyAuditMapper.selectList(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getManyMarkId, id)
        );
    }

    @Override
    public int confirmAudit(Integer id) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }

        List<ManyAuditEntity> list = manyAuditMapper.selectList(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getManyMarkId, manyMarkEntity.getId())
                        .eq(ManyAuditEntity::getIsSubmit, 0)
                        .eq(ManyAuditEntity::getIsLose, 0)
        );
        /*if (CollectionUtils.isEmpty(list)) {
            return 0;
        }*/
        for (ManyAuditEntity auditEntity : list) {
            auditEntity.setAuditState(1);
            manyAuditMapper.updateById(auditEntity);
        }

        // 修改任务状态
        manyMarkEntity.setTaskState(5); //TODO 任务设置成待审核
        return manyMarkMapper.updateById(manyMarkEntity);
    }

    @Override
    public int examineTaskShift(TaskShiftRequest request) {
        long currentTime1 = System.currentTimeMillis();
        // 获取当前的用户id
        ManyAuditEntity currentAuditEntity = manyAuditMapper.selectById(request.getCurrentUserId());
        if (ObjectUtils.isEmpty(currentAuditEntity)) {
            return 0;
        }


        // 修改文件表
        List<ManyFileEntity> list = manyFileMapper.selectList(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, currentAuditEntity.getManyMarkId())
                        .eq(ManyFileEntity::getUserId, currentAuditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getAuditUserId, currentAuditEntity.getUserId())
        );

        long currentTime2 = System.currentTimeMillis();

        System.out.println("执行时间1：" + (currentTime2 - currentTime1) + "毫秒");

        if (CollectionUtils.isEmpty(list)) {
            return 1;
        }

        // 异步批量更新文件表
        for (ManyFileEntity entity : list) {
            entity.setAuditUserId(request.getShiftId());
        }
        manyFileMapper.updateBatchByIds(list);

        // 修改待转交
        currentAuditEntity.setUserId(request.getShiftId());
        long currentTime3 = System.currentTimeMillis();
        System.out.println("执行时间2：" + (currentTime3 - currentTime2) + "毫秒");
        return manyAuditMapper.updateById(currentAuditEntity);
    }


    @Override
    public int distributionExamine(Integer taskId) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }

        // 判断当前任务是否分配过
        List<ManyAuditEntity> list = manyAuditMapper.selectList(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getManyMarkId, taskId)
        );

        if (!CollectionUtils.isEmpty(list)) {
            return 1;
        }


        // 取出审核人员
        List<Integer> examineUserList = teamUserMapper.selectList(
                new LambdaQueryWrapper<TeamUserEntity>()
                        .eq(TeamUserEntity::getTeamId, manyMarkEntity.getAuditTeamId())
        ).stream().map(TeamUserEntity::getUserId).collect(Collectors.toList());

        if (CollectionUtils.isEmpty(examineUserList)) {
            return 0;
        }

        // 取出标注人员
        List<ManyAssignEntity> manyAssignList = manyAssignMapper.selectList(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, manyMarkEntity.getId())
                        .eq(ManyAssignEntity::getIsLose, 0)
        );

        if (CollectionUtils.isEmpty(manyAssignList)) {
            return 0;
        }

        allocationTask(examineUserList, manyAssignList, manyMarkEntity);
        return 1;
    }


    // 分配任务

    /**
     * @param examineUserList 审核人员
     * @param manyAssignList  任务集合
     */
    @Async
    public void allocationTask(List<Integer> examineUserList, List<ManyAssignEntity> manyAssignList, ManyMarkEntity manyMarkEntity) {
        int userIndex = 0;
        for (ManyAssignEntity manyAssignEntity : manyAssignList) {
            int userId = examineUserList.get(userIndex);
            // 这里可以添加具体的分配逻辑，例如将任务与用户关联存储到数据库等
            /* System.out.println("将任务 " + manyAssignEntity.getUserId() + " 分配给审核人员 " + userId);*/
            userIndex = (userIndex + 1) % examineUserList.size();

            // 记录数据库
            ManyAuditEntity auditEntity = getManyAuditEntity(manyMarkEntity, manyAssignEntity, userId);
            manyAuditMapper.insert(auditEntity);

            // 更改任务文件表
            updateManyFile(auditEntity);

           /* // 修改任务状态
            manyMarkEntity.setTaskState(5); //TODO 任务设置成待审核
            manyMarkMapper.updateById(manyMarkEntity);*/
        }
    }

    // 记录数据库
    private static ManyAuditEntity getManyAuditEntity(ManyMarkEntity manyMarkEntity, ManyAssignEntity manyAssignEntity, int userId) {
        String fileIds = manyAssignEntity.getAssignFileIds();
        ManyAuditEntity auditEntity = new ManyAuditEntity();
        auditEntity.setManyMarkId(manyMarkEntity.getId());
        auditEntity.setSonId(manyMarkEntity.getSonId());
        auditEntity.setUserId(userId);
        auditEntity.setMarkUserId(manyAssignEntity.getUserId());
        auditEntity.setAuditFileIds(fileIds);
        auditEntity.setYesExamine(0);
        auditEntity.setNoExamine(fileIds.split(",").length);
        auditEntity.setProgress(0);
        auditEntity.setAuditState(7);
        auditEntity.setCreateTime(new Date());
        auditEntity.setIsLose(0);
        auditEntity.setIsSubmit(0);
        return auditEntity;
    }

    // 更改任务文件表
    private void updateManyFile(ManyAuditEntity auditEntity) {
        // 将文件 ID 字符串拆分为整数列表
        String[] roleIdStrs = auditEntity.getAuditFileIds().split(",");
        List<Integer> fileIds = new ArrayList<>(roleIdStrs.length);
        for (String roleIdStr : roleIdStrs) {
            fileIds.add(Integer.parseInt(roleIdStr));
        }

        // 批量查询符合条件的 ManyFileEntity 对象
        List<ManyFileEntity> entities = manyFileMapper.selectBatchByTaskAndFileIds(
                auditEntity.getManyMarkId(),
                fileIds,
                auditEntity.getMarkUserId()
        );

        // 更新查询到的对象的审核人员 ID
        for (ManyFileEntity entity : entities) {
            entity.setAuditUserId(auditEntity.getUserId());
        }

        // 批量更新
        if (!entities.isEmpty()) {
            manyFileMapper.updateBatchByIds(entities);
        }
    }


    @Override
    public boolean submitTaskShift(Integer taskId) {
        return manyAssignMapper.selectCount(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, taskId)
                        .eq(ManyAssignEntity::getIsLose, 0)
                        .eq(ManyAssignEntity::getIsSubmit, 0)
        ) != 0;
    }

    @Override
    public boolean isExamineStatus(ExamineReturnRequest request) {
        List<ManyAuditEntity> manyAuditList = null;
        if (CollectionUtils.isEmpty(request.getIds())) {
            manyAuditList = manyAuditMapper.selectList(new LambdaQueryWrapper<ManyAuditEntity>()
                    .eq(ManyAuditEntity::getManyMarkId, request.getTaskId())
                    .orderByAsc(ManyAuditEntity::getId));
        } else {
            manyAuditList = manyAuditMapper.selectBatchIds(request.getIds());
        }
        for (ManyAuditEntity auditEntity : manyAuditList) {
            if (auditEntity.getAuditState() != 4) {
                return false;
            }
        }
        return true;
    }


    @Override
    public int examineReturn(ExamineReturnRequest request) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(request.getTaskId());
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }

        List<ManyAuditEntity> manyAuditList = null;
        List<ManyAuditEntity> filterManyAuditList = null;
        if (request.getIds() == null) { // 退回所有
            manyAuditList = manyAuditMapper.selectList(new LambdaQueryWrapper<ManyAuditEntity>()
                    .eq(ManyAuditEntity::getManyMarkId, manyMarkEntity.getId())
                    .orderByAsc(ManyAuditEntity::getId));
            // 过滤数据
            filterManyAuditList = filterDuplicateUserMarkUserCombinations(manyAuditList);
        } else {  // 批量打回 或者 单个
            manyAuditList = manyAuditMapper.selectManyAuditByIds(request.getIds());
            // 过滤数据
            filterManyAuditList = filterDuplicateUserMarkUserCombinations(manyAuditList);
        }

        for (ManyAuditEntity manyAuditEntity : filterManyAuditList) {
            updateAuditEntity(manyAuditEntity);
            updateFiles(manyMarkEntity, manyAuditEntity);
        }
        // 批量进行修改审核员数据
        manyAuditMapper.batchUpdateManyAuditEntities(filterManyAuditList);
      /*  // 最终修改任务状态
        manyMarkEntity.setTaskState(5);
        return manyMarkMapper.updateById(manyMarkEntity);*/
        return 1;
    }


    @Override
    public boolean isExamineSubmit(Integer taskId) {
        return selectAllSubmit(taskId) == 0;

    }

    @Override
    public int approved(Integer taskId) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (manyMarkEntity == null) {
            return 0;
        }

        // 如果所有审核人员全部提交，更改任务状态
        if (selectAllSubmit(taskId) == 0) {
            // 证明所有审核人员任务都已经提交
            manyMarkEntity.setTaskState(7);
            return manyMarkMapper.updateById(manyMarkEntity);
        }
        return 0;
    }

    private int selectAllSubmit(Integer taskId) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (manyMarkEntity == null) {
            return -1;
        }

        return manyAuditMapper.selectCount(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getManyMarkId, manyMarkEntity.getId())
                        .eq(ManyAuditEntity::getIsLose, 0)
                        .eq(ManyAuditEntity::getIsSubmit, 0)
        );
    }


    private void updateFiles(ManyMarkEntity manyMarkEntity, ManyAuditEntity manyAuditEntity) {
        // 获取所有文件
        List<ManyFileEntity> entityList = manyFileMapper.selectList(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, manyMarkEntity.getId())
                        .eq(ManyFileEntity::getUserId, manyAuditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getAuditUserId, manyAuditEntity.getUserId())
        );
        for (ManyFileEntity manyFileEntity : entityList) {
            manyFileEntity.setNotPassMessage("");
            manyFileEntity.setIsApprove(0);
        }
        // 批量修改文件数据
        manyFileMapper.updateBatchByIds(entityList);
    }


    private void updateAuditEntity(ManyAuditEntity manyAuditEntity) {
        manyAuditEntity.setAuditState(8);
        manyAuditEntity.setYesExamine(0);
        manyAuditEntity.setNoExamine(getFileCount(manyAuditEntity.getAuditFileIds()));
        manyAuditEntity.setProgress(0);
        manyAuditEntity.setIsLose(0);
        manyAuditEntity.setIsSubmit(0);
    }

    // 获取所有文件数量
    private int getFileCount(String auditFileIds) {
        if (ObjectUtils.isEmpty(auditFileIds)) {
            return 0;
        }
        return auditFileIds.split(",").length;
    }


    @Transient
    private List<ManyAuditEntity> filterDuplicateUserMarkUserCombinations(List<ManyAuditEntity> entities) {
        List<ManyAuditEntity> result = new ArrayList<>();
        Set<String> userMarkUserCombinations = new HashSet<>();
        List<Integer> idsToDelete = new ArrayList<>();  // 要删除的审核人员数据

        for (ManyAuditEntity entity : entities) {
            String combination = entity.getUserId() + "-" + entity.getMarkUserId();
            if (userMarkUserCombinations.add(combination)) {
                result.add(entity);
            } else {
                idsToDelete.add(entity.getId());
            }
        }

        if (!idsToDelete.isEmpty()) {
            // 执行删除
            manyAuditMapper.deleteBatchIds(idsToDelete);
        }
        return result;
    }


}
