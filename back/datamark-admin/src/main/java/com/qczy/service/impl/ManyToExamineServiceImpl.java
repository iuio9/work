package com.qczy.service.impl;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.mapper.*;

import com.qczy.model.entity.*;
import com.qczy.service.ManyToExamineService;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.MyProgressUtils;
import com.qczy.utils.StringUtils;
import org.apache.ibatis.annotations.Many;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/12 10:52
 * @Description:
 */
@Service
public class ManyToExamineServiceImpl implements ManyToExamineService {

    @Autowired
    private ManyMarkMapper manyMarkMapper;
    @Autowired
    private ManyFileMapper manyFileMapper;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private ManyAuditMapper manyAuditMapper;


    // 计算进度
    private ManyAuditEntity updateManyAuditProgress(ManyAuditEntity manyAuditEntity) {
        // 文件总数量
        Integer fileCount = manyAuditEntity.getAuditFileIds().split(",").length;
        // 当前未审核数量
        Integer noExamineCount = manyFileMapper.selectCount(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId,manyAuditEntity.getManyMarkId())
                        .eq(ManyFileEntity::getAuditUserId, manyAuditEntity.getUserId())
                        .eq(ManyFileEntity::getUserId, manyAuditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getIsApprove, 0)
        );
        manyAuditEntity.setYesExamine(fileCount - noExamineCount);
        manyAuditEntity.setNoExamine(noExamineCount);
        // 计算进度
        manyAuditEntity.setProgress(MyProgressUtils.calculateCount((fileCount - noExamineCount), fileCount));

        // 如果进度为100，修改状态（审核已完成）
        if (manyAuditEntity.getProgress() == 100) {
            manyAuditEntity.setAuditState(3);
        }
        return manyAuditEntity;
    }
 /*   private ManyAuditEntity updateManyAuditProgress(ManyAuditEntity manyAuditEntity) {
        // 1. 处理auditFileIds，获取去重且非空的文件ID集合
        String auditFileIds = manyAuditEntity.getAuditFileIds();
        if (StringUtils.isEmpty(auditFileIds)) {
            manyAuditEntity.setYesExamine(0);
            manyAuditEntity.setNoExamine(0);
            manyAuditEntity.setProgress(0);
            return manyAuditEntity;
        }

        String[] fileIdArray = auditFileIds.split(",");
        Set<String> uniqueFileIds = new HashSet<>();
        for (String fileId : fileIdArray) {
            if (StringUtils.isNotBlank(fileId)) {
                uniqueFileIds.add(fileId.trim());
            }
        }
        int fileCount = uniqueFileIds.size();

        // 全是无效ID时直接返回，避免后续无效计算
        if (fileCount == 0) {
            manyAuditEntity.setYesExamine(0);
            manyAuditEntity.setNoExamine(0);
            manyAuditEntity.setProgress(0);
            return manyAuditEntity;
        }

        // 2. 查询未审核数量，限制文件ID范围
        Integer noExamineCount = manyFileMapper.selectCount(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, manyAuditEntity.getManyMarkId())
                        .eq(ManyFileEntity::getAuditUserId, manyAuditEntity.getUserId())
                        .eq(ManyFileEntity::getUserId, manyAuditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getIsApprove, 0)
                        .in(ManyFileEntity::getFileId, uniqueFileIds)
        );
        // 处理可能的null值
        noExamineCount = (noExamineCount == null) ? 0 : noExamineCount;

        // 3. 计算已审核数量（确保非负）
        int yesExamine = Math.max(0, fileCount - noExamineCount);
        manyAuditEntity.setYesExamine(yesExamine);
        manyAuditEntity.setNoExamine(noExamineCount);

        // 4. 计算进度（确保在0-100之间）
        int progress = MyProgressUtils.calculateCount(yesExamine, fileCount);
        progress = Math.min(100, Math.max(0, progress));
        manyAuditEntity.setProgress(progress);

        // 5. 进度100%时更新状态
        if (progress == 100) {
            manyAuditEntity.setAuditState(3);
        }

        return manyAuditEntity;
    }*/

    @Override
    public int isApprove(ManyFileEntity manyFileEntity) {
        //
        ManyFileEntity entity = manyFileMapper.selectOne(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, manyFileEntity.getTaskId())
                        .eq(ManyFileEntity::getFileId, manyFileEntity.getFileId())
        );
        if (ObjectUtils.isEmpty(entity)) {
            return 0;
        }
        entity.setIsApprove(manyFileEntity.getIsApprove());
        if (manyFileEntity.getIsApprove() == 1) {
            entity.setNotPassMessage("");
        } else {
            entity.setNotPassMessage(manyFileEntity.getNotPassMessage());
        }
        int result = manyFileMapper.updateById(entity);
        // 如果为第一次，则更改任务状态
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(entity.getTaskId());
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        if (manyMarkEntity.getTaskState() == 5) {
            // 改为审核中
            manyMarkEntity.setTaskState(6);
            manyMarkMapper.updateById(manyMarkEntity);
        }

        // 更改审核员审核进度
        ManyAuditEntity auditEntity = manyAuditMapper.selectOne(new LambdaQueryWrapper<ManyAuditEntity>()
                .eq(ManyAuditEntity::getManyMarkId, entity.getTaskId())
                .eq(ManyAuditEntity::getMarkUserId, entity.getUserId())
                .eq(ManyAuditEntity::getUserId, entity.getAuditUserId())
                .eq(ManyAuditEntity::getIsLose, 0));

        if (ObjectUtils.isEmpty(auditEntity)) {
            return 0;
        }
        // 如果当前状态为 待审核， 更改为审核中
        if (auditEntity.getAuditState() == 1 || auditEntity.getAuditState() == 7 || auditEntity.getAuditState() == 8) {
            auditEntity.setAuditState(2);
        }
        // 计算进度,并修改
        manyAuditMapper.updateById(updateManyAuditProgress(auditEntity));
        return result;
    }

    @Override
   /* public int verifyComplete(Integer taskId, Integer verifyState) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, manyMarkEntity.getSonId()));
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return 0;
        }
        if (verifyState == 2) { // 只保存验收通过的数据
            // 收集未通过的数据，删除标注信息
            List<ManyFileEntity> noVerifyList = manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, taskId)
                            .eq(ManyFileEntity::getIsApprove, 0)
                            .or()
                            .eq(ManyFileEntity::getIsApprove, 2)
            );
            for (ManyFileEntity entity : noVerifyList) {
                Integer fileId = entity.getFileId();
                markInfoMapper.delete(new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getFileId, fileId));
            }
            // 更改数据集进度
            // 总文件大小
            int fileCount = dataSonEntity.getFileIds().split(",").length;
            // 当前标注的数量
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            int progress = MyProgressUtils.calculateCount(count, fileCount);
            dataSonEntity.setStatus(progress + "% " + ("(" + count + "/" + fileCount + ")"));
        }

        // 更改任务状态
        manyMarkEntity.setTaskState(8); //TODO 任务结束
        int result = manyMarkMapper.updateById(manyMarkEntity);
        // 更改数据集状态
        dataSonEntity.setIsMany(0);
        dataSonMapper.updateById(dataSonEntity);
        deleteManyFile(taskId);
        return result;
    }*/


    public int verifyComplete(Integer taskId, Integer verifyState) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }

        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, manyMarkEntity.getSonId())
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return 0;
        }

        if (verifyState == 2) { // 只保存验收通过的数据
            // 收集未通过的数据，删除标注信息
            List<ManyFileEntity> noVerifyList = manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, taskId)
                            .eq(ManyFileEntity::getIsApprove, 0)
                            .or()
                            .eq(ManyFileEntity::getIsApprove, 2)
            );
            for (ManyFileEntity entity : noVerifyList) {
                Integer fileId = entity.getFileId();
                markInfoMapper.delete(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getFileId, fileId)
                );
            }

            // 更改数据集进度（核心修复部分）
            // 1. 处理fileIds，获取去重、非空的有效文件总数
            String fileIds = dataSonEntity.getFileIds();
            Set<String> validFileIds = new HashSet<>();
            if (StringUtils.isNotBlank(fileIds)) {
                String[] fileIdArray = fileIds.split(",");
                for (String fileId : fileIdArray) {
                    if (StringUtils.isNotBlank(fileId)) { // 过滤空值和空白
                        validFileIds.add(fileId.trim());
                    }
                }
            }
            int fileCount = validFileIds.size(); // 有效文件总数（去重后）

            // 2. 统计标注数时，限制在有效文件范围内
            Integer count = 0;
            if (!validFileIds.isEmpty()) {
                count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                                .in(MarkInfoEntity::getFileId, validFileIds) // 关键：限定文件范围
                );
                // 处理可能的null（如果mapper返回Integer）
                count = (count == null) ? 0 : count;
            }

            // 3. 计算进度，强制限制在0-100%
            int progress = 0;
            if (fileCount > 0) {
                // 确保已标注数不超过总文件数（防止进度超100%）
                int actualCount = Math.min(count, fileCount);
                progress = MyProgressUtils.calculateCount(actualCount, fileCount);
                progress = Math.min(100, Math.max(0, progress)); // 强制边界
            }

            // 4. 更新状态文本
            dataSonEntity.setStatus(progress + "% (" + count + "/" + fileCount + ")");
        }

        // 更改任务状态
        manyMarkEntity.setTaskState(8); // 任务结束
        int result = manyMarkMapper.updateById(manyMarkEntity);

        // 更改数据集状态
        dataSonEntity.setIsMany(0);
        dataSonMapper.updateById(dataSonEntity);

        deleteManyFile(taskId);
        return result;
    }


    @Async
    public void deleteManyFile(Integer taskId) {
        manyFileMapper.deleteBatchIds(
                manyFileMapper.selectList(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                ).stream().map(ManyFileEntity::getId).collect(Collectors.toList())
        );
    }

    @Override
    public int returnTask(Integer taskId, Integer returnState, Integer id) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        switch (returnState) {
            case 1:
                newAllocation(taskId, manyMarkEntity, 0, id);
                break;
            case 2:
                newAllocation(taskId, manyMarkEntity, 2, id);
                break;
            case 3:
                newAllocation(taskId, manyMarkEntity, 3, id);
                break;
        }

        // 更改任务状态
        //manyMarkEntity.setTaskState(3);
        return manyMarkMapper.updateById(manyMarkEntity);
    }

    @Override
    public int remainingApprove(Integer taskId, Integer id) {
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);
        if (ObjectUtils.isEmpty(auditEntity)) {
            return 0;
        }
        List<ManyFileEntity> data = manyFileMapper.selectList(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, taskId)
                        .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                        .eq(ManyFileEntity::getIsApprove, 0)
        );

        // 修改查询出的记录的 isApprove 属性
        for (ManyFileEntity entity : data) {
            entity.setIsApprove(1);
        }


        // 批量更新 ManyFileEntity 记录
        if (!data.isEmpty()) {
            manyFileMapper.updateBatchByIds(data);
        }


        // 计算进度,并修改
        manyAuditMapper.updateById(updateManyAuditProgress(auditEntity));

        return 1;
    }


    @Override
    public boolean isRemaining(Integer taskId, Integer id) {
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return false;
        }
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);
        if (ObjectUtils.isEmpty(auditEntity)) {
            return false;
        }

        return manyFileMapper.selectCount(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, manyMarkEntity.getId())
                        .eq(ManyFileEntity::getAuditUserId, userId)
                        .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getIsApprove, 0)
        ) != 0;
    }

    @Override
    public boolean returnTaskState(Integer taskId, Integer returnState, Integer id) {
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);
        if (ObjectUtils.isEmpty(auditEntity)) {
            return false;
        }

        switch (returnState) {
            case 1:
                return manyFileMapper.selectCount(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                                .eq(ManyFileEntity::getIsApprove, 0)
                                .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                                .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                ) != 0;

            case 2:
                return manyFileMapper.selectCount(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                                .eq(ManyFileEntity::getIsApprove, 2)
                                .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                                .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                ) != 0;
            case 3:
                return (manyFileMapper.selectCount(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                                .eq(ManyFileEntity::getIsApprove, 0)
                                .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                                .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                )) + (manyFileMapper.selectCount(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                                .eq(ManyFileEntity::getIsApprove, 2)
                                .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                                .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                )) != 0;

        }
        return false;
    }


    @Override
    public int submitTask(Integer id) {
        // 获取当前用户
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            return 0;
        }
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(manyAssignEntity.getManyMarkId());
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        manyAssignEntity.setIsSubmit(1);
        manyAssignEntity.setUserState(7);
        manyAssignMapper.updateById(manyAssignEntity);
        // 这里判断是否分配审核了，如果分配了，则修改状态
        ManyAuditEntity auditEntity = manyAuditMapper.selectOne(
                new LambdaQueryWrapper<ManyAuditEntity>()
                        .eq(ManyAuditEntity::getMarkUserId, manyAssignEntity.getUserId())
                        .eq(ManyAuditEntity::getManyMarkId, manyAssignEntity.getManyMarkId())
                        .eq(ManyAuditEntity::getIsLose, 0)
                        .eq(ManyAuditEntity::getIsSubmit, 0)
        );
        if (!ObjectUtils.isEmpty(auditEntity)) {
            auditEntity.setAuditState(1);
            manyAuditMapper.updateById(auditEntity);
        } else {
            // 当前分配的总人数
            Integer count = manyAssignMapper.selectCount(
                    new LambdaQueryWrapper<ManyAssignEntity>()
                            .eq(ManyAssignEntity::getManyMarkId, manyMarkEntity.getId())
                            .eq(ManyAssignEntity::getIsLose, 0)
            );
            // 当前是否所有人都提交
            Integer submitCount = manyAssignMapper.selectCount(
                    new LambdaQueryWrapper<ManyAssignEntity>()
                            .eq(ManyAssignEntity::getManyMarkId, manyMarkEntity.getId())
                            .eq(ManyAssignEntity::getUserState, 7)
                            .eq(ManyAssignEntity::getIsSubmit, 1)
                            .eq(ManyAssignEntity::getIsLose, 0)
            );

            if (count.equals(submitCount)) {
                // 更改任务状态
                manyMarkEntity.setTaskState(4);
                manyMarkMapper.updateById(manyMarkEntity);
                return 1;
            } else {
                if (manyMarkEntity.getTaskState() != 3) {
                    manyMarkEntity.setTaskState(3);
                    manyMarkMapper.updateById(manyMarkEntity);
                }
                return 1;
            }

        }

        // 更改任务状态
        manyMarkEntity.setTaskState(6);
        manyMarkMapper.updateById(manyMarkEntity);
        return 1;
    }

    @Override
    public String submitTaskPrompt(Integer id) {
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            return "当前任务不存在, 不可提交！";
        }
        // 获取所有文件
        String fileIds = manyAssignEntity.getAssignFileIds();
        int noMark = fileMapper.selectFileAndlabelNoMarkCount(fileIds);
        if (noMark == 0) {
            return "提交后不可在进行标注！确认要提交吗？";
        }
        return "当前您有" + noMark + "项未标注, 提交后不可在进行标注！确认要提交吗？";
    }


    public void newAllocation(Integer taskId, ManyMarkEntity manyMarkEntity, Integer state, Integer id) {
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);

        if (ObjectUtils.isEmpty(auditEntity)) {
            return;
        }
        Integer markUserId = auditEntity.getMarkUserId();

        // 更改标注员的数据状态
        // TODO 更改之前的标注员的数据状态
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectOne(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, taskId)
                        .eq(ManyAssignEntity::getUserId, markUserId)
                        .eq(ManyAssignEntity::getIsLose, 0)
        );
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            return;
        }

        manyAssignEntity.setUserState(6);
        manyAssignEntity.setIsLose(1);
        manyAssignMapper.updateById(manyAssignEntity);
        String fileIds = getFileIds(auditEntity,state);
        addManyAssign(manyMarkEntity, markUserId, fileIds);
        // TODO 更改之前审核员的数据状态
        auditEntity.setAuditState(5); // 已驳回
        auditEntity.setIsLose(1);
        manyAuditMapper.updateById(auditEntity);
        addManyAudit(manyMarkEntity, auditEntity.getUserId(), fileIds, markUserId);

    }

    public List<Integer> getUserList(Integer taskId, Integer state) {
        if (state == 3) { // 未验收 + 验收不通过
            return manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, taskId)
                            .eq(ManyFileEntity::getIsApprove, 0)
                            .or()
                            .eq(ManyFileEntity::getIsApprove, 2)
            ).stream().map(ManyFileEntity::getUserId).distinct().collect(Collectors.toList());
        } else {
            return manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, taskId)
                            .eq(ManyFileEntity::getIsApprove, state)
            ).stream().map(ManyFileEntity::getUserId).distinct().collect(Collectors.toList());
        }
    }

    public String getFileIds(ManyAuditEntity auditEntity, Integer state) {
        List<ManyFileEntity> list = null;
        if (state == 3) { // 未验收 + 验收不通过
            list = manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, auditEntity.getManyMarkId())
                            .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                            .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                            // 使用 and 方法包裹 or 条件组，确保在基础条件范围内进行或逻辑
                            .and(wrapper -> wrapper
                                    .eq(ManyFileEntity::getIsApprove, 0)
                                    .or()
                                    .eq(ManyFileEntity::getIsApprove, 2)
                            )
            );
        } else {
            list = manyFileMapper.selectList(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, auditEntity.getManyMarkId())
                            .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                            .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                            .eq(ManyFileEntity::getIsApprove, state)
            );
        }

        if (CollectionUtils.isEmpty(list)) {
            return null;
        }
        // 提取 fileId 并使用逗号分割
        return list.stream()
                .map(ManyFileEntity::getFileId)
                .map(String::valueOf)
                .collect(Collectors.joining(","));
    }

    public void addManyAssign(ManyMarkEntity manyMarkEntity, Integer userId, String fileIds) {

        if (ObjectUtils.isEmpty(fileIds)) {
            return;
        }
        // 将 fileIds 字符串分割成数组
        String[] fileIdArray = fileIds.split(",");
        List<Integer> fileIdList = new ArrayList<>();
        for (String fileIdStr : fileIdArray) {
            fileIdList.add(Integer.parseInt(fileIdStr));
        }

        // 批量查询符合条件的文件实体
        List<ManyFileEntity> entityList = manyFileMapper.selectList(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .in(ManyFileEntity::getFileId, fileIdList)
                        .eq(ManyFileEntity::getUserId, userId)
        );

        if (CollectionUtils.isEmpty(entityList)) {
            return;
        }

        // 修改文件状态
        for (ManyFileEntity entity : entityList) {
            entity.setIsApprove(0);
        }

        // 批量更新文件状态
        manyFileMapper.updateBatchByIds(entityList);

        ManyAssignEntity manyAssignEntity = new ManyAssignEntity();
        manyAssignEntity.setManyMarkId(manyMarkEntity.getId());
        manyAssignEntity.setSonId(manyMarkEntity.getSonId());
        // 文件总数量
        int fileNum = fileIds.split(",").length;
        // 查看未标注的数量
        int noMarkNum = fileMapper.selectFileAndlabelNoMarkCount(fileIds);
        manyAssignEntity.setYesMark(fileNum - noMarkNum);
        manyAssignEntity.setNoMark(noMarkNum);
        // 计算进度
        manyAssignEntity.setProgress(MyProgressUtils.calculateCount(manyAssignEntity.getYesMark(), fileNum) + "");
        manyAssignEntity.setAssignFileIds(fileIds);
        manyAssignEntity.setUserId(userId);
        manyAssignEntity.setUserState(1);
        manyAssignEntity.setCreateTime(new Date());
        manyAssignEntity.setIsLose(0);
        manyAssignEntity.setIsSubmit(0);
        manyAssignMapper.insert(manyAssignEntity);
    }


    public void addManyAudit(ManyMarkEntity manyMarkEntity, Integer userId, String fileIds, Integer markUserId) {
        ManyAuditEntity manyAuditEntity = new ManyAuditEntity();
        manyAuditEntity.setManyMarkId(manyMarkEntity.getId());
        manyAuditEntity.setSonId(manyMarkEntity.getSonId());
        manyAuditEntity.setAuditFileIds(fileIds);
        manyAuditEntity.setUserId(userId);
        manyAuditEntity.setMarkUserId(markUserId);
        manyAuditEntity.setYesExamine(0);
        manyAuditEntity.setNoExamine(fileIds.split(",").length);
        manyAuditEntity.setProgress(0);
        manyAuditEntity.setCreateTime(new Date());
        manyAuditEntity.setAuditState(7);
        manyAuditEntity.setIsSubmit(0);
        manyAuditEntity.setIsLose(0);
        manyAuditMapper.insert(manyAuditEntity);
    }


}
