package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.excel.ExcelUtils;
import com.qczy.common.exception.GlobalExceptionHandler;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.AssocDataSetRequest;
import com.qczy.model.request.CopyLabelGroupRequest;
import com.qczy.model.request.FileIdsRequest;
import com.qczy.model.response.LabelGroupAndLabelResponse;
import com.qczy.service.LabelGroupService;
import com.qczy.service.LabelService;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 14:46
 * @Description:
 */
@Service
public class LabelGroupServiceImpl extends ServiceImpl<LabelGroupMapper, LabelGroupEntity> implements LabelGroupService {


    @Autowired
    private LabelGroupMapper labelGroupMapper;

    @Autowired
    private LabelMapper labelMapper;

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private LabelService labelService;

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public LabelGroupEntity selectLabelGroupById(Integer id) {
        return labelGroupMapper.selectLabelGroupById(id);
    }

    /**
     * 查询【请填写功能名称】列表
     *
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<LabelGroupEntity> selectLabelGroupList(Page<LabelGroupEntity> pageParam, LabelGroupEntity labelGroup) {
        return labelGroupMapper.selectLabelGroupList(pageParam, labelGroup);
    }

    /**
     * 新增【请填写功能名称】
     *
     * @param labelGroup 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertLabelGroup(LabelGroupEntity labelGroup) {
        labelGroup.setCreateTime(new Date());
        return labelGroupMapper.insertLabelGroup(labelGroup);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param labelGroup 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateLabelGroup(LabelGroupEntity labelGroup) {
        labelGroup.setUpdateTime(new Date());
        LabelGroupEntity labelGroupEntity = labelGroupMapper.selectById(labelGroup.getId());
        List<LabelEntity> labelEntityList = labelMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<LabelEntity>()
                        .eq(LabelEntity::getLabelGroupId, labelGroupEntity.getId())
        );

        if (!CollectionUtils.isEmpty(labelEntityList)) {
            String newGroupName = labelGroup.getEnglishLabelGroupName();
            String oldGroupName = labelGroupEntity.getEnglishLabelGroupName();

            if (StringUtils.isEmpty(oldGroupName)) {
                for (LabelEntity labelEntity : labelEntityList) {
                    if (StringUtils.isEmpty(labelEntity.getEnglishLabelName())) {
                        labelEntity.setEnglishLabelName(newGroupName);
                    } else if (!labelEntity.getEnglishLabelName().startsWith(newGroupName)) {
                        labelEntity.setEnglishLabelName(newGroupName + "_" + labelEntity.getEnglishLabelName());
                    }
                }
            } else {
                for (LabelEntity labelEntity : labelEntityList) {
                    if (labelEntity.getEnglishLabelName().startsWith(oldGroupName)) {
                        // 替换前缀
                        labelEntity.setEnglishLabelName(newGroupName + labelEntity.getEnglishLabelName().substring(oldGroupName.length()));
                    } else {
                        labelEntity.setEnglishLabelName(newGroupName + "_" + labelEntity.getLabelName());
                    }
                }
            }

            // 统一设置 onlyId 并批量更新
            for (LabelEntity labelEntity : labelEntityList) {
                labelEntity.setOnlyId(labelEntity.getEnglishLabelName());
            }
            labelMapper.updateBatchById(labelEntityList);
        }

        return labelGroupMapper.updateLabelGroup(labelGroup);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteLabelGroupByIds(int[] ids) {

        for (int groupId : ids) {
            /**
             *  删除当前标标签组的所有标签
             */
            List<LabelEntity> list = labelMapper.selectList(new LambdaQueryWrapper<LabelEntity>().eq(LabelEntity::getLabelGroupId, groupId));
            if (!CollectionUtils.isEmpty(list)) {
                List<Integer> labelIds = list.stream().map(LabelEntity::getId).collect(Collectors.toList());
                labelMapper.deleteBatchIds(labelIds);

                /**
                 *  删除跟数据集绑定的所有标签
                 */
                for (Integer labelId : labelIds) {
                    List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(new LambdaQueryWrapper<DataSonLabelEntity>().eq(DataSonLabelEntity::getLabelId, labelId));
                    if (!CollectionUtils.isEmpty(dataSonLabelEntityList)) {
                        dataSonLabelMapper.deleteBatchIds(dataSonLabelEntityList.stream().map(DataSonLabelEntity::getId).collect(Collectors.toList()));
                    }
                }
            }


        }


        return labelGroupMapper.deleteLabelGroupByIds(ids);
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteLabelGroupById(Integer id) {
        return labelGroupMapper.deleteById(id);
    }

    @Override
    public List<LabelGroupAndLabelResponse> selectLabelList() {
        List<LabelGroupEntity> labelGroupEntityList = labelGroupMapper.selectList(null);
        if (CollectionUtils.isEmpty(labelGroupEntityList)) {
            return null;
        }
        List<LabelGroupAndLabelResponse> labelGroupAndLabelResponseList = new ArrayList<>();
        for (LabelGroupEntity labelGroupEntity : labelGroupEntityList) {
            LabelGroupAndLabelResponse labelGroupAndLabelResponse = new LabelGroupAndLabelResponse();
            BeanUtils.copyProperties(labelGroupEntity, labelGroupAndLabelResponse);
            labelGroupAndLabelResponse.setList(
                    labelMapper.selectList(
                            new LambdaQueryWrapper<LabelEntity>()
                                    .eq(LabelEntity::getLabelGroupId, labelGroupEntity.getId())
                    )
            );
            labelGroupAndLabelResponseList.add(labelGroupAndLabelResponse);
        }
        return labelGroupAndLabelResponseList;
    }

    @Override
    public List<Integer> getSonIdByLabelGroupIds(String sonId) {

        DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(sonId);

        // 判断标签是什么
        String tagSelectionMode = dataSonEntity.getTagSelectionMode();
        if (StringUtils.isEmpty(tagSelectionMode)) {
            return null;
        } else {
            List<DataSonLabelEntity> dataSonLabelEntityList = null;
            List<Integer> labelIds = null;
            switch (tagSelectionMode) {
                case "group":
                    // 查询所有标签
                    dataSonLabelEntityList = dataSonLabelMapper.selectList(
                            new LambdaQueryWrapper<DataSonLabelEntity>()
                                    .eq(DataSonLabelEntity::getSonId, sonId));

                    labelIds = new ArrayList<>();
                    for (DataSonLabelEntity dataSonLabelEntity : dataSonLabelEntityList) {
                        LabelEntity labelEntity = labelMapper.selectById(dataSonLabelEntity.getLabelId());
                        labelIds.add(labelEntity.getLabelGroupId());
                    }
                    return labelIds.stream().distinct().collect(Collectors.toList());
                case "single":
                    // 查询所有标签
                    dataSonLabelEntityList = dataSonLabelMapper.selectList(
                            new LambdaQueryWrapper<DataSonLabelEntity>()
                                    .eq(DataSonLabelEntity::getSonId, sonId));

                    labelIds = new ArrayList<>();
                    for (DataSonLabelEntity dataSonLabelEntity : dataSonLabelEntityList) {
                        LabelEntity labelEntity = labelMapper.selectById(dataSonLabelEntity.getLabelId());
                        labelIds.add(labelEntity.getId());
                    }
                    return labelIds.stream().distinct().collect(Collectors.toList());
            }
            return null;
        }
    }


    @Override
    public void addDataSonAndLabelGroup(String sonId, String groupIds) {
        String[] ids = groupIds.split(",");

        for (String labelGroupId : ids) {
            List<LabelEntity> labelEntityList = labelMapper.selectList(new LambdaQueryWrapper<LabelEntity>()
                    .eq(LabelEntity::getLabelGroupId, labelGroupId));

            if (CollectionUtils.isEmpty(labelEntityList)) {
                continue;
            }

            // 开始新增
            for (LabelEntity entity : labelEntityList) {
                DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
                dataSonLabelEntity.setSonId(sonId);
                dataSonLabelEntity.setLabelId(entity.getId());
                dataSonLabelEntity.setLabelCount(0);
                dataSonLabelMapper.insert(dataSonLabelEntity);
            }

        }
    }

    @Override
    public void addDataSonAndLabelIds(String sonId, String tagIds) {
        String[] ids = tagIds.split(",");
        for (String labelId : ids) {
            DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
            dataSonLabelEntity.setSonId(sonId);
            dataSonLabelEntity.setLabelId(Integer.parseInt(labelId));
            dataSonLabelEntity.setLabelCount(0);
            dataSonLabelMapper.insert(dataSonLabelEntity);
        }
    }

    @Override
    public int copyLabelGroup(CopyLabelGroupRequest request) {

        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, request.getSonId())
        );

        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("数据集对象不存在！");
        }

        DataSonEntity copyDataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, request.getCopySonId())
        );
        if (ObjectUtils.isEmpty(copyDataSonEntity)) {
            throw new RuntimeException("拷贝的数据集对象不存在！");
        }

        List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, dataSonEntity.getSonId())
        );


        List<DataSonLabelEntity> distinctDataSonLabelList = dataSonLabelEntityList.stream().distinct().collect(Collectors.toList());


        List<DataSonLabelEntity> copyDataSonLabelEntityList = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, copyDataSonEntity.getSonId())
        );

        List<DataSonLabelEntity> distinctCopyDataSonLabelList = copyDataSonLabelEntityList.stream().distinct().collect(Collectors.toList());
        if (CollectionUtils.isEmpty(distinctCopyDataSonLabelList)) {
            return 1;
        }

        List<Integer> copyLabelId = distinctCopyDataSonLabelList.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());

        if (!CollectionUtils.isEmpty(distinctDataSonLabelList)) {
            List<Integer> idList = distinctDataSonLabelList.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());
            if (request.getBusinessType() == 1) { // 全部覆盖
                dataSonLabelMapper.deleteBatchIds(idList);
            } else {  //累加，但是需要删除重复数据
                List<Integer> combinedList = new ArrayList<>();
                combinedList.addAll(idList);
                combinedList.addAll(copyLabelId);
                List<Integer> duplicates = combinedList.stream()
                        .filter(i -> combinedList.indexOf(i) != combinedList.lastIndexOf(i))
                        .distinct()
                        .collect(Collectors.toList());
                System.out.println(duplicates);
                dataSonLabelMapper.deleteBatchIds(duplicates);
            }
        }


        // 最后统一新增数据
        for (Integer labelId : copyLabelId) {
            LabelEntity labelEntity = labelMapper.selectById(labelId);
            if (ObjectUtils.isEmpty(labelEntity)) {
                continue;
            }
            DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
            dataSonLabelEntity.setLabelCount(0);
            dataSonLabelEntity.setSonId(dataSonEntity.getSonId());
            dataSonLabelEntity.setLabelId(labelId);
            dataSonLabelMapper.insert(dataSonLabelEntity);
        }
        return 1;
    }


    @Override
    public int assocDataSet(AssocDataSetRequest request) {
        LabelGroupEntity labelGroupEntity = labelGroupMapper.selectById(request.getLabelGroupId());
        if (ObjectUtils.isEmpty(labelGroupEntity)) {
            return 1;
        }

        List<LabelEntity> labelEntityList = labelMapper.selectList(
                new LambdaQueryWrapper<LabelEntity>()
                        .eq(LabelEntity::getLabelGroupId, labelGroupEntity.getId())
        );

        if (CollectionUtils.isEmpty(labelEntityList)) {
            return 1;
        }

        for (String sonId : request.getDataSetIdList()) {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, sonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                continue;
            }

            for (LabelEntity labelEntity : labelEntityList) {
                Integer count = dataSonLabelMapper.selectCount(
                        new LambdaQueryWrapper<DataSonLabelEntity>()
                                .eq(DataSonLabelEntity::getLabelId, labelEntity.getId())
                                .eq(DataSonLabelEntity::getSonId, sonId)
                );
                if (count == 0) {
                    DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
                    dataSonLabelEntity.setSonId(sonId);
                    dataSonLabelEntity.setLabelId(labelEntity.getId());
                    dataSonLabelEntity.setLabelCount(0);
                    dataSonLabelMapper.insert(dataSonLabelEntity);
                }

            }
        }
        return 1;
    }

    @Override
    public String getDataSonLabelStatus(String sonId) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, sonId)
        );

        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("数据集对象不存在！");
        }

        List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>().eq(
                        DataSonLabelEntity::getSonId, dataSonEntity.getSonId()
                )
        );

        if (CollectionUtils.isEmpty(dataSonLabelEntityList)) {
            return "0个标签组、0个自定义标签，共计0条标签数据";
        }

        List<Integer> collect = dataSonLabelEntityList.stream()
                .map(DataSonLabelEntity::getLabelId)
                .collect(Collectors.toSet())
                .stream()
                .sorted()
                .collect(Collectors.toList());
        // 数据集组 id
        Set<Integer> ids = new HashSet<>();
        List<Integer> toRemove = new ArrayList<>();
        int countSum = collect.size();
        int i = 0;
        for (Integer labelId : collect) {
            LabelEntity labelEntity = labelMapper.selectById(labelId);
            if (!ObjectUtils.isEmpty(labelEntity) && labelEntity.getLabelGroupId() != 0) {
                ids.add(labelEntity.getLabelGroupId());
                toRemove.add(labelId);
                i++;
            }
        }
        collect.removeAll(toRemove);

        int customize = countSum - i;
        //  int sumCount = ids.size() + customize;

        return ids.size() + "个标签组、" + customize + "个自定义标签，" + "共计" + countSum + "条标签数据";
    }

    @Autowired
    private TempFileMapper tempFileMapper;

    @Override
    public int importLabel(FileIdsRequest request) {
        try {
            for (int fileId : request.getFileIds()) {
                TempFileEntity tempFileEntity = tempFileMapper.selectById(fileId);
                File file = new File(tempFileEntity.getFdTempPath());
                // 读取 Excel 文件内容
                List<LabelEntity> labelEntities = ExcelUtils.readFile(file, LabelEntity.class);
                // 检查列表是否为空
                if (ObjectUtils.isEmpty(labelEntities)) {
                    continue;
                }

                // 标签组信息
                List<LabelGroupEntity> labelGroupList = new ArrayList<>();

                // 获取所有标签组名称
                for (LabelEntity labelEntity : labelEntities) {
                    LabelGroupEntity labelGroupEntity = new LabelGroupEntity();
                    labelGroupEntity.setLabelGroupName(labelEntity.getLabelName().split("/")[0].trim());
                    labelGroupEntity.setEnglishLabelGroupName(labelEntity.getOnlyId().split("/")[0].trim());
                    labelGroupList.add(labelGroupEntity);
                }
                // 进行去重操作
                List<LabelGroupEntity> groupList = labelGroupList.stream().distinct().collect(Collectors.toList());


                LabelGroupEntity lg = new LabelGroupEntity();
                for (LabelGroupEntity labelGroupEntity : groupList) {
                    // 判断标签组是否存在
                    List<LabelGroupEntity> labelGroupEntityList = labelGroupMapper.selectList(
                            new LambdaQueryWrapper<LabelGroupEntity>()
                                    .eq(LabelGroupEntity::getLabelGroupName, labelGroupEntity.getLabelGroupName())
                                    .orderByDesc(LabelGroupEntity::getCreateTime)
                    );
                    // 存在
                    if (!CollectionUtils.isEmpty(labelGroupEntityList)) {
                        lg.setId(labelGroupEntityList.get(0).getId());
                        lg.setEnglishLabelGroupName(labelGroupEntityList.get(0).getEnglishLabelGroupName());
                        if (StringUtils.isEmpty(lg.getEnglishLabelGroupName())) {
                            lg.setEnglishLabelGroupName(labelGroupEntity.getEnglishLabelGroupName());
                            // 修改标签组
                            labelGroupMapper.updateById(lg);
                            // 异步执行之前的标签组，加上英文
                            updateOldLabel(lg);
                        }
                    } else { // 不存在
                        LabelGroupEntity entity = new LabelGroupEntity();
                        entity.setLabelGroupName(labelGroupEntity.getLabelGroupName());
                        entity.setEnglishLabelGroupName(labelGroupEntity.getEnglishLabelGroupName());
                        if (!insertLabelGroupEntity(entity)) {
                            continue;
                        }
                        lg = entity;
                    }

                    // 获取当前这个标签组下的数据
                    List<LabelEntity> labelEntityList = new ArrayList<>();
                    for (LabelEntity labelEntity : labelEntities) {
                        if (labelGroupEntity.getLabelGroupName().equals(labelEntity.getLabelName().split("/")[0].trim())) {
                            labelEntityList.add(labelEntity);
                        }
                    }
                    if (ObjectUtils.isEmpty(labelEntityList)) {
                        continue;
                    }

                    // 新增所有标签
                    insertLabels(labelEntityList, lg);
                }
            }
            return 1;
        } catch (Exception e) {
            // 打印异常信息，便于调试
            e.printStackTrace();
            return 0;
        }
    }


    public void updateOldLabel(LabelGroupEntity lg) {
        List<LabelEntity> labelEntityList = labelMapper.selectList(new LambdaQueryWrapper<LabelEntity>().eq(LabelEntity::getLabelGroupId, lg.getId()));
        if (CollectionUtils.isEmpty(labelEntityList)) {
            return;
        }
        for (LabelEntity labelEntity : labelEntityList) {
            labelEntity.setOnlyId(lg.getLabelGroupName() + "_" + labelEntity.getLabelName());
            labelEntity.setEnglishLabelName(labelEntity.getOnlyId());
        }
        labelMapper.updateBatchById(labelEntityList);

    }


    private String extractLabelGroupName(LabelEntity labelEntity) {
        if (ObjectUtils.isEmpty(labelEntity) || StringUtils.isEmpty(labelEntity.getLabelName())) {
            return null;
        }
        int index = labelEntity.getLabelName().indexOf('/');
        return index != -1 ? labelEntity.getLabelName().substring(0, index) : null;
    }

    private boolean insertLabelGroupEntity(LabelGroupEntity labelGroupEntity) {
        int result = labelGroupMapper.insert(labelGroupEntity);
        return result == 1;
    }

    private boolean insertLabels(List<LabelEntity> labelEntities, LabelGroupEntity lg) {
        for (LabelEntity entity : labelEntities) {
            if (ObjectUtils.isEmpty(entity)) {
                continue;
            }
            LabelEntity newLabelEntity = createNewLabelEntity(entity, lg);
            if (newLabelEntity != null) {
                labelMapper.insert(newLabelEntity);
            }

        }
        return true;
    }

    private LabelEntity createNewLabelEntity(LabelEntity entity, LabelGroupEntity lg) {
        LabelEntity newLabelEntity = new LabelEntity();
        newLabelEntity.setLabelGroupId(lg.getId());

        // 处理标签唯一编号
        String onlyId = entity.getOnlyId();
        if (!StringUtils.isEmpty(onlyId)) {
            int onlyIdIndex = onlyId.indexOf('/');
            if (onlyIdIndex != -1) {
                newLabelEntity.setOnlyId(lg.getEnglishLabelGroupName() + "_" + onlyId.substring(onlyIdIndex + 1).trim());
                newLabelEntity.setEnglishLabelName(newLabelEntity.getOnlyId());
            }

        }

        // 处理标签名称
        String labelName = entity.getLabelName();
        if (!StringUtils.isEmpty(labelName)) {
            int labelNameIndex = labelName.indexOf('/');
            if (labelNameIndex != -1) {
                newLabelEntity.setLabelName(labelName.substring(labelNameIndex + 1).trim());
                // 判断标签是否重复
                if (labelService.isExistLabelName(lg.getId(), newLabelEntity.getLabelName(), null)) {
                    return null;
                }
            }
        }


        // 处理标签颜色
        String labelColor = entity.getLabelColor();
        if (!StringUtils.isEmpty(labelColor)) {
            // 进行转换
            newLabelEntity.setLabelColor(convertRGBStringToHex(labelColor));
        } else {
            // 分配给随机一个颜色，不允许重复
            newLabelEntity.setLabelColor(getRandomUniqueColor());

        }


        return newLabelEntity;
    }

    // 已使用的颜色列表
    private static List<String> usedColors = new ArrayList<>();
    private static final Random random = new Random();


    private static String getRandomUniqueColor() {
        String color;
        do {
            int r = random.nextInt(256);
            int g = random.nextInt(256);
            int b = random.nextInt(256);
            color = String.format("#%02X%02X%02X", r, g, b);
        } while (usedColors.contains(color));
        usedColors.add(color);
        return color;
    }


    public static String convertRGBStringToHex(String rgbString) {
        // 提取 R、G、B 的值
        int r = Integer.parseInt(rgbString.split("R=")[1].split(",")[0]);
        int g = Integer.parseInt(rgbString.split("G=")[1].split(",")[0]);
        int b = Integer.parseInt(rgbString.split("B=")[1].split("\\)")[0]);

        // 转换为十六进制
        String hexR = String.format("%02X", r);
        String hexG = String.format("%02X", g);
        String hexB = String.format("%02X", b);

        return "#" + hexR + hexG + hexB;
    }


    @Override
    public boolean isExistLabelGroupName(Integer groupId, String groupName) {
        LambdaQueryWrapper<LabelGroupEntity> queryWrapper = new LambdaQueryWrapper<LabelGroupEntity>()
                .eq(LabelGroupEntity::getLabelGroupName, groupName.trim());

        if (groupId != null) {
            queryWrapper.ne(LabelGroupEntity::getId, groupId);
        }
        Integer count = labelGroupMapper.selectCount(queryWrapper);
        return count != 0;
    }


    @Override
    public boolean isExistLabelEnglishGroupName(Integer groupId, String englishGroupName) {
        LambdaQueryWrapper<LabelGroupEntity> queryWrapper = new LambdaQueryWrapper<LabelGroupEntity>()
                .eq(LabelGroupEntity::getEnglishLabelGroupName, englishGroupName.trim());

        if (groupId != null) {
            queryWrapper.ne(LabelGroupEntity::getId, groupId);
        }
        Integer count = labelGroupMapper.selectCount(queryWrapper);
        return count != 0;
    }


}
