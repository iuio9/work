package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.support.SFunction;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.ManyMarkTaskRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.ManyCreateListResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.model.response.TeamUserResponse;
import com.qczy.service.ManyMarkService;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.MyProgressUtils;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 16:31
 * @Description:
 */
@Service
public class ManyMarkServiceImpl extends ServiceImpl<ManyMarkMapper, ManyMarkEntity>
        implements ManyMarkService {

    @Autowired
    private ManyMarkMapper manyMarkMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private TeamMapper teamMapper;
    @Autowired
    private TeamUserMapper teamUserMapper;
    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private ManyFileMapper manyFileMapper;
    @Autowired
    private ManyAuditMapper manyAuditMapper;
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;


    @Override
    public IPage<ManyCreateListResponse> getMyCreateTaskList(Page<ManyCreateListResponse> pageParam) {
        // 获取当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        return manyMarkMapper.getMyCreateTaskList(pageParam, userId);
    }

    @Override
    public int endTask(Integer taskId) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }
        manyMarkEntity.setTaskState(4);
        // 更改所有用户的状态
        List<ManyAssignEntity> manyAssignList = manyAssignMapper.selectList(
                new LambdaQueryWrapper<ManyAssignEntity>().eq(ManyAssignEntity::getManyMarkId, taskId)
        );
        for (ManyAssignEntity manyAssignEntity : manyAssignList) {
            manyAssignEntity.setUserState(4);
            manyAssignEntity.setIsSubmit(1);
            manyAssignMapper.updateById(manyAssignEntity);
        }
  /*      // 更改数据集状态
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, manyMarkEntity.getSonId()));
        dataSonEntity.setIsMany(0);
        dataSonMapper.updateById(dataSonEntity);*/
        return manyMarkMapper.updateById(manyMarkEntity);
    }

    @Override
    public int deleteTask(Integer taskId) {
        List<Integer> assignIds = manyAssignMapper.selectList(
                        new LambdaQueryWrapper<ManyAssignEntity>()
                                .eq(ManyAssignEntity::getManyMarkId, taskId)
                ).stream()
                .map(ManyAssignEntity::getId)
                .collect(Collectors.toList());
        if (!CollectionUtils.isEmpty(assignIds)) manyAssignMapper.deleteBatchIds(assignIds);


        List<Integer> auditIds = manyAuditMapper.selectList(
                        new LambdaQueryWrapper<ManyAuditEntity>()
                                .eq(ManyAuditEntity::getManyMarkId, taskId)
                ).stream()
                .map(ManyAuditEntity::getId)
                .collect(Collectors.toList());
        if (!CollectionUtils.isEmpty(auditIds)) manyAuditMapper.deleteBatchIds(auditIds);

        List<Integer> manyFileIds = manyFileMapper.selectList(
                        new LambdaQueryWrapper<ManyFileEntity>()
                                .eq(ManyFileEntity::getTaskId, taskId)
                ).stream()
                .map(ManyFileEntity::getId)
                .collect(Collectors.toList());
        if (!CollectionUtils.isEmpty(manyFileIds)) manyFileMapper.deleteBatchIds(manyFileIds);

        // 修改数据集状态
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(taskId);
        if (!ObjectUtils.isEmpty(manyMarkEntity)) {
            if (!StringUtils.isEmpty(manyMarkEntity.getSonId())) {
                DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                        new LambdaQueryWrapper<DataSonEntity>()
                                .eq(DataSonEntity::getSonId, manyMarkEntity.getSonId())
                );
                if (dataSonEntity != null) {
                    dataSonEntity.setIsMany(0);
                    dataSonMapper.updateById(dataSonEntity);
                }
            }
        }
        return manyMarkMapper.deleteById(taskId);
    }


    @Override
    public IPage<ManyReceiveListResponse> getMyReceiveList(Page<ManyReceiveListResponse> pageParam) {
        // 获取当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        return manyAssignMapper.getMyReceiveList(pageParam, userId);
    }

    @Override
    public boolean isRelayed(TaskShiftRequest request) {
        ManyAssignEntity currentManyAssignEntity = manyAssignMapper.selectById(request.getCurrentUserId());
        if (ObjectUtils.isEmpty(currentManyAssignEntity)) {
            return false;
        }
        // 如果当前人出现过打回的数据，就不许再次转交
        /* if (currentManyAssignEntity.getUserState() == 6)*/
        List<ManyAssignEntity> manyAssignEntities = manyAssignMapper.selectList(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, currentManyAssignEntity.getManyMarkId())
                        .eq(ManyAssignEntity::getUserId, currentManyAssignEntity.getUserId())
        );
        if (!CollectionUtils.isEmpty(manyAssignEntities)) {
            for (ManyAssignEntity manyAssignEntity : manyAssignEntities) {
                if (manyAssignEntity.getUserState() == 6) {
                    return false;
                }
            }
        }

        ManyAssignEntity shiftManyAssignEntity = manyAssignMapper.selectOne(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, currentManyAssignEntity.getManyMarkId())
                        .eq(ManyAssignEntity::getUserId, request.getShiftId())
                        .eq(ManyAssignEntity::getIsLose, 0)
        );
        if (ObjectUtils.isEmpty(shiftManyAssignEntity)) {
            return false;
        }

        switch (shiftManyAssignEntity.getUserState()) {
            case 1:
            case 2:
                return true;
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
                return false;

        }
        return false;
    }

    @Override
    @Transactional
    public int taskShift(TaskShiftRequest request) {
        ManyAssignEntity currentManyAssignEntity = manyAssignMapper.selectById(request.getCurrentUserId());
        if (ObjectUtils.isEmpty(currentManyAssignEntity)) {
            return 0;
        }


        ManyAssignEntity shiftManyAssignEntity = manyAssignMapper.selectOne(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, currentManyAssignEntity.getManyMarkId())
                        .eq(ManyAssignEntity::getUserId, request.getShiftId())
                        .eq(ManyAssignEntity::getIsLose, 0)
        );
        if (ObjectUtils.isEmpty(shiftManyAssignEntity)) {
            return 0;
        }

        // 获取当前用户的文件
        String currentFiles = currentManyAssignEntity.getAssignFileIds();
        // 获取要移交人的文件
        String shiftFiles = shiftManyAssignEntity.getAssignFileIds();
        // 新的文件集
        String newFiles = currentFiles + "," + shiftFiles;
        // 新的文件数量
        int newFileSize = newFiles.split(",").length;
        // 老的数据集的标注、未标注数量
        Integer oldYesMark = currentManyAssignEntity.getYesMark();
        Integer oldNoMark = currentManyAssignEntity.getNoMark();
        // 更改需要转交的用户状态
        currentManyAssignEntity.setUserState(5);
        currentManyAssignEntity.setYesMark(0);
        currentManyAssignEntity.setNoMark(0);
        currentManyAssignEntity.setProgress("0");
        currentManyAssignEntity.setAssignFileIds("");
        currentManyAssignEntity.setIsLose(1);
        manyAssignMapper.updateById(currentManyAssignEntity);
        // 更改转交人的状态、信息
        shiftManyAssignEntity.setYesMark(shiftManyAssignEntity.getYesMark() + oldYesMark);
        shiftManyAssignEntity.setNoMark(shiftManyAssignEntity.getNoMark() + oldNoMark);
        shiftManyAssignEntity.setAssignFileIds(newFiles);

        // 计算进度
        int progress = MyProgressUtils.calculateCount(shiftManyAssignEntity.getYesMark() + oldYesMark, newFileSize);
        shiftManyAssignEntity.setProgress(progress + "");
        int result = manyAssignMapper.updateById(shiftManyAssignEntity);

        // 删除
        for (String fileIdStr : currentFiles.split(",")) {
            manyFileMapper.delete(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, currentManyAssignEntity.getManyMarkId())
                            .eq(ManyFileEntity::getUserId, currentManyAssignEntity.getUserId())
                            .eq(ManyFileEntity::getFileId, Integer.valueOf(fileIdStr))
            );
        }

        // 一边删除 、 一边新增
        for (String fileIdStr : currentFiles.split(",")) {
            // 删除
            manyFileMapper.delete(
                    new LambdaQueryWrapper<ManyFileEntity>()
                            .eq(ManyFileEntity::getTaskId, currentManyAssignEntity.getManyMarkId())
                            .eq(ManyFileEntity::getUserId, currentManyAssignEntity.getUserId())
                            .eq(ManyFileEntity::getFileId, Integer.valueOf(fileIdStr))
            );
            // 新增
            manyFileMapper.insert(new ManyFileEntity(shiftManyAssignEntity.getManyMarkId(), shiftManyAssignEntity.getUserId(),
                    Integer.valueOf(fileIdStr), 0));
        }

        return result;

    }

    @Override
    public List<TeamUserResponse> getByTaskIdTeamList(Integer taskId, Integer teamType) {
        if (teamType == null) {
            return null;
        }
        if (teamType == 1) {
            return teamMapper.getByTaskIdTeamList(taskId);
        } else if (teamType == 2) {
            return teamMapper.getByTaskIdTeamList1(taskId);
        }
        return null;
    }

    @Override
    public int endUserTask(Integer id) {
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            return 0;
        }
        manyAssignEntity.setUserState(4);
        manyAssignEntity.setIsSubmit(1);
        int result = manyAssignMapper.updateById(manyAssignEntity);
        // 如果这个团队的所有任务都已经为提交任务，则修改任务状态
        List<ManyAssignEntity> manyAssignEntityList = manyAssignMapper.selectList(
                new LambdaQueryWrapper<ManyAssignEntity>()
                        .eq(ManyAssignEntity::getManyMarkId, manyAssignEntity.getManyMarkId())
                        .eq(ManyAssignEntity::getIsLose, 0)
                        .eq(ManyAssignEntity::getIsSubmit, 0)
        );
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(manyAssignEntity.getManyMarkId());
        // 判断当前任务的状态，如果是开始审核了，则不修改任务状态
        if (manyMarkEntity.getTaskState() >= 5) {
            // TODO 更改当前审核员的状态，允许修改
            ManyAuditEntity auditEntity = manyAuditMapper.selectOne(
                    new LambdaQueryWrapper<ManyAuditEntity>()
                            .eq(ManyAuditEntity::getManyMarkId, manyMarkEntity.getId())
                            .eq(ManyAuditEntity::getMarkUserId, manyAssignEntity.getUserId())
                            .eq(ManyAuditEntity::getIsSubmit, 0)
                            .eq(ManyAuditEntity::getIsLose, 0)
            );
            if (ObjectUtils.isEmpty(auditEntity)) {
                return 0;
            }
            // 修改状态
            auditEntity.setAuditState(1);
            result = manyAuditMapper.updateById(auditEntity);
            return result;
        }
        // TODO 当前状态还处于标注状态
        if (CollectionUtils.isEmpty(manyAssignEntityList)) {
            // 修改任务状态
            manyMarkEntity.setTaskState(4);
            result = manyMarkMapper.updateById(manyMarkEntity);
        }

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
    @Transactional
    public int withdraw(DeleteRequest request) {
        // 获取当前的任务id
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(request.getTaskId());
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            return 0;
        }


        List<Integer> markIds = new ArrayList<>();
        // 查询哪些图片符合条件
        for (Integer markId : request.getIds()) {
            MarkInfoEntity markInfoEntity = markInfoMapper.selectById(markId);
            if (!ObjectUtils.isEmpty(markInfoEntity)) {
                markIds.add(markInfoEntity.getId());
            }
        }

        if (CollectionUtils.isEmpty(markIds)) {
            return 0;
        }

        // 删除标注信息
/*        List<Integer> markIds = Arrays.stream(request.getIds())
                // 将 IntStream 中的每个元素装箱为 Integer 类型
                .boxed()
                // 将流中的元素收集到一个 List 中
                .collect(Collectors.toList());*/

        int result = markInfoMapper.deleteBatchIds(markIds);

        // 已标注
        int yesMarkSize = manyAssignEntity.getYesMark() - markIds.size();
        // 未标注
        int noMarkSize = manyAssignEntity.getNoMark() + markIds.size();
        // 总数量
        int fileSize = manyAssignEntity.getAssignFileIds().split(",").length;
        // 计算进度
        int progress = MyProgressUtils.calculateCount(yesMarkSize, fileSize);


        // 更改当前任务的状态
        manyAssignEntity.setYesMark(yesMarkSize);
        manyAssignEntity.setNoMark(noMarkSize);
        manyAssignEntity.setProgress(progress + "");
        manyAssignMapper.updateById(manyAssignEntity);

        // 修改数据集，更改进度
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(manyAssignEntity.getManyMarkId());
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return 0;
        }

        // 记录标注进度
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, manyMarkEntity.getSonId())
        );

        // 计算进度
        if (!ObjectUtils.isEmpty(dataSonEntity)) {
            String[] fileIds = dataSonEntity.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            int progress1 = MyProgressUtils.calculateCount(count, fileIds.length);
            dataSonEntity.setStatus(progress1 + "% " + ("(" + count + "/" + fileIds.length + ")"));
            dataSonMapper.updateById(dataSonEntity);
        }

        return result;
    }


    public String allocationNum(String sonId, String teamId) {
        if (isDataSet(sonId)) {
            return "当前数据集数据不符合要求, 无法分配样本。 ";
        }
        Map<String, Object> map = fileMap(sonId);
        if (CollectionUtils.isEmpty(map)) {
            return null;
        }
        int fileSize = 0;
        Object sizeObj = map.get("size");
        if (sizeObj != null) {
            fileSize = (Integer) sizeObj;
        }

        List<TeamUserEntity> teamUserList = teamUserMapper.selectList(
                new LambdaQueryWrapper<TeamUserEntity>()
                        .eq(TeamUserEntity::getTeamId, teamId)
        );
        int teamSize = teamUserList.size();

        if (teamSize == 0) {
            return "数据集需标注样本总数" + fileSize + "个, 团队人数为 0, 无法分配样本。";
        }

        List<UserEntity> userList = new ArrayList<>();
        for (TeamUserEntity teamUserEntity : teamUserList) {
            UserEntity user = userMapper.selectById(teamUserEntity.getUserId());
            if (!ObjectUtils.isEmpty(user)) {
                userList.add(user);
            }
        }

        Map<Integer, Integer> allocationMap = new HashMap<>();
        for (UserEntity user : userList) {
            allocationMap.put(user.getId(), 0);
        }

        int index = 0;
        for (int i = 0; i < fileSize; i++) {
            UserEntity user = userList.get(index);
            allocationMap.put(user.getId(), allocationMap.get(user.getId()) + 1);
            index = (index + 1) % userList.size();
        }

        StringBuilder result = new StringBuilder();
        result.append("数据集需标注样本总数").append(fileSize).append("个。\n");
        result.append("具体分配情况如下：\n");
        for (Map.Entry<Integer, Integer> entry : allocationMap.entrySet()) {
            int allocationCount = entry.getValue();
            if (allocationCount > 0) {
                UserEntity user = getUserById(entry.getKey(), userList);
                if (user != null && user.getNickName() != null && !user.getNickName().isEmpty()) {
                    result.append("用户昵称: ").append(user.getNickName());
                }
                result.append(", 分配样本数: ").append(allocationCount).append("个\n");
            }
        }

        return result.toString();
    }

    private UserEntity getUserById(Integer id, List<UserEntity> userList) {
        for (UserEntity user : userList) {
            if (user.getId().equals(id)) {
                return user;
            }
        }
        return null;
    }

    @Override
    public boolean isDataSet(String sonId) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        // 数据集不存在，或者文件为空！
        if (ObjectUtils.isEmpty(dataSonEntity) || StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            return true;
        }
        Map<String, Object> map = fileMap(dataSonEntity.getSonId());
        return CollectionUtils.isEmpty(map) || (map.get("size") == null || (Integer) map.get("size") == 0);
    }


    /**
     * 当前数据集符合条件的文件
     *
     * @param sonId 数据集id
     * @return
     */
    private Map<String, Object> fileMap(String sonId) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return null;
        }
        Map<String, Object> map = new HashMap<>();
        // 获取当前数据集未标注的所有文件
        List<DataDetailsResponse> list = fileMapper.selectFileAndlabelNoMarkNoPage(dataSonEntity.getFileIds());
        String fileIds = list.stream()
                .map(DataDetailsResponse::getFileId)
                .map(Object::toString)
                .collect(Collectors.joining(","));
        map.put("fileIds", fileIds);
        map.put("size", StringUtils.isEmpty(fileIds) ? 0 : fileIds.split(",").length);
        return map;
    }


    @Override
    public int countBySonIdTask(String sonId) {
        return manyMarkMapper.selectCount(
                new LambdaQueryWrapper<ManyMarkEntity>().eq(ManyMarkEntity::getSonId, sonId)
        ) +
                modelAssessConfigMapper.selectCount(
                        new LambdaQueryWrapper<ModelAssessConfigEntity>().eq(ModelAssessConfigEntity::getSonId, sonId)
                );
    }

    @Override
    public int countBySonIdsTask(String groupId) {
        // 查询当前数据集组
        List<DataSonEntity> sonEntityList = dataSonMapper.selectList(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getFatherId, groupId)
        );

        if (CollectionUtils.isEmpty(sonEntityList)) {
            return 0;
        }
        int num = 0;
        for (DataSonEntity dataSonEntity : sonEntityList) {
            num += (
                    manyMarkMapper.selectCount(
                            new LambdaQueryWrapper<ManyMarkEntity>().eq(ManyMarkEntity::getSonId, dataSonEntity.getSonId())
                    ) +
                            modelAssessConfigMapper.selectCount(
                                    new LambdaQueryWrapper<ModelAssessConfigEntity>().eq(ModelAssessConfigEntity::getSonId, dataSonEntity.getSonId())
                            )
            );
        }
        return num;
    }

    @Override
    public int addManyMarkTask(ManyMarkTaskRequest request) {
        // 获取当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, request.getSonId())
        );

        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("数据集不存在！");
        }
        if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            throw new RuntimeException("文件数量不能为空！");
        }
        // 状态设置为：正在分配中
        request.setTaskState(1);
        request.setUserId(userId);
        request.setCreateTime(new Date());
        int result = manyMarkMapper.insert(request);
        // 分配任务
        allocationTask(request.getTeamId(), dataSonEntity, request);
        return result;
    }


    /**
     * 分配任务
     */
    @Async
    public void allocationTask(Integer teamId, DataSonEntity dataSonEntity, ManyMarkTaskRequest manyMarkTaskRequest) {
        TeamEntity teamEntity = teamMapper.selectById(teamId);
        if (ObjectUtils.isEmpty(teamEntity)) {
            throw new RuntimeException("多人标注团队不存在！");
        }
        List<TeamUserEntity> teamUsers = teamUserMapper.selectList(
                new LambdaQueryWrapper<TeamUserEntity>()
                        .eq(TeamUserEntity::getTeamId, teamId)
        );

        // 文件数据
        // TODO 1.处理有标注信息的
        Map<String, Object> map = fileMap(dataSonEntity.getSonId());
        if (CollectionUtils.isEmpty(map)) {
            return;
        }
        String fileIds = (String) map.get("fileIds");

        // 开始分配
        List<String> data = splitIds(fileIds, teamUsers.size());
        System.out.println("---------------------" + data.size());

        List<ManyAssignEntity> manyAssignEntities = new ArrayList<>();
        List<ManyFileEntity> manyFileEntities = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            TeamUserEntity teamUserEntity = teamUsers.get(i);
            // 每个用户分配的信息
            String allocationFileIds = data.get(i);
            String[] fileIdArray = allocationFileIds.split(",");

            ManyAssignEntity manyAssignEntity = new ManyAssignEntity();
            manyAssignEntity.setManyMarkId(manyMarkTaskRequest.getId());
            manyAssignEntity.setSonId(dataSonEntity.getSonId());
            manyAssignEntity.setAssignFileIds(allocationFileIds);
            manyAssignEntity.setUserId(teamUserEntity.getUserId());
            manyAssignEntity.setNoMark(fileIdArray.length);
            manyAssignEntity.setYesMark(0);
            manyAssignEntity.setProgress("0");
            manyAssignEntity.setUserState(1);
            manyAssignEntity.setCreateTime(new Date());
            manyAssignEntity.setIsLose(0);
            manyAssignEntity.setIsSubmit(0);
            manyAssignEntities.add(manyAssignEntity);

            // 具体分配的文件信息
            for (String fileId : fileIdArray) {
                manyFileEntities.add(new ManyFileEntity(manyMarkTaskRequest.getId(), teamUserEntity.getUserId(),
                        Integer.parseInt(fileId), 0));
            }
        }

        // 批量插入 ManyAssignEntity
        if (!manyAssignEntities.isEmpty()) {
            manyAssignMapper.insertBatch(manyAssignEntities);
        }

        // 批量插入 ManyFileEntity
        if (!manyFileEntities.isEmpty()) {
            manyFileMapper.insertBatch(manyFileEntities);
        }

        // TODO 2.再次更改任务状态，
        manyMarkTaskRequest.setTaskState(2);
        manyMarkMapper.updateById(manyMarkTaskRequest);

        // 把数据集状态设置成不可操作
        dataSonEntity.setIsMany(1);
        dataSonMapper.updateById(dataSonEntity);
    }


    public static List<String> splitIds(String idString, int parts) {
        // 先将字符串按逗号分割成数组
        String[] ids = idString.split(",");
        int total = ids.length;

        // 如果人数超出文件数量，将部分数调整为文件数量
        if (parts > total) {
            parts = total;
        }

        // 计算平均每份的数量
        int average = total / parts;
        // 计算余数
        int remainder = total % parts;
        List<String> result = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < parts; i++) {
            int currentCount = average;
            if (i < remainder) {
                // 余数部分依次分配到前面的部分
                currentCount++;
            }
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < currentCount; j++) {
                sb.append(ids[start + j]);
                if (j < currentCount - 1) {
                    sb.append(",");
                }
            }
            result.add(sb.toString());
            start += currentCount;
        }
        return result;
    }


    public static void main(String[] args) {
        /*StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 500; i++) {
            sb.append(i);
            if (i < 500) {
                sb.append(",");
            }
        }*/
        String idString = "1,2";
        List<String> parts = splitIds(idString, 4);
        for (int i = 0; i < parts.size(); i++) {
            System.out.println(parts.get(i));
            System.out.println("第 " + (i + 1) + " 个用户分配的 id 数量: " + parts.get(i).split(",").length);
        }

        // System.out.println(19 / 2);

    }
}
