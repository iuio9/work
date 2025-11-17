package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.AddDataSetLabelRequest;
import com.qczy.model.request.DeleteDataSetLabelRequest;
import com.qczy.model.request.LabelEntityRequest;
import com.qczy.model.response.DataSetLabelResponse;
import com.qczy.model.response.GroupLabelResponse;
import com.qczy.service.LabelService;
import com.qczy.utils.JsonUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 15:20
 * @Description:
 */
@Service
public class LabelServiceImpl implements LabelService {

    @Autowired
    private LabelMapper labelMapper;

    @Autowired
    private LabelGroupMapper labelGroupMapper;

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private MarkInfoMapper markInfoMapper;

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public LabelEntity selectLabelById(Integer id) {
        return labelMapper.selectLabelById(id);
    }

    /**
     * 查询【请填写功能名称】列表
     *
     * @param label 【请填写功能名称】
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<LabelEntity> selectLabelList(Page<LabelEntity> pageParam, Integer labelGroupId, LabelEntity label) {
        return labelMapper.selectLabelList(pageParam, labelGroupId, label);
    }

    /**
     * 新增【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertLabel(LabelEntity label, LabelGroupEntity entity) {
        label.setCreateTime(new Date());
        label.setLabelName(label.getLabelName().trim());
        String e1 = entity.getEnglishLabelGroupName();
        String e2 = label.getEnglishLabelName();
        if (e2.startsWith(e1)) {
            label.setEnglishLabelName(label.getEnglishLabelName().trim());
        } else {
            label.setEnglishLabelName(entity.getEnglishLabelGroupName() + "_" + label.getEnglishLabelName().trim());
        }
        label.setOnlyId(label.getEnglishLabelName());
        return labelMapper.insertLabel(label);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateLabel(LabelEntity label, LabelGroupEntity entity) {
        label.setUpdateTime(new Date());
        label.setLabelName(label.getLabelName().trim());
        if (entity != null) {
            String e1 = entity.getEnglishLabelGroupName();
            String e2 = label.getEnglishLabelName();
            if (e2.startsWith(e1)) {
                label.setEnglishLabelName(label.getEnglishLabelName().trim());
            } else {
                label.setEnglishLabelName(entity.getEnglishLabelGroupName() + "_" + label.getEnglishLabelName().trim());
            }
        } else {
            label.setEnglishLabelName(label.getEnglishLabelName().trim());
        }

        label.setOnlyId(label.getEnglishLabelName());
        return labelMapper.updateLabel(label);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteLabelByIds(int[] ids) {
        for (int id : ids) {
            List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(new LambdaQueryWrapper<DataSonLabelEntity>().eq(DataSonLabelEntity::getLabelId, id));
            if (!CollectionUtils.isEmpty(dataSonLabelEntityList)) {
                dataSonLabelMapper.deleteBatchIds(dataSonLabelEntityList.stream().map(DataSonLabelEntity::getId).collect(Collectors.toList()));
            }
        }
        return labelMapper.deleteLabelByIds(ids);
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteLabelById(Integer id) {
        return labelMapper.deleteLabelById(id);
    }

    @Override
    public List<GroupLabelResponse> selectGroupLabel() {
        List<GroupLabelResponse> list = new ArrayList<>();
        List<LabelGroupEntity> labelGroupEntityList = labelGroupMapper.selectList(
                new LambdaQueryWrapper<LabelGroupEntity>().orderByDesc(LabelGroupEntity::getId)
        );
        if (CollectionUtils.isEmpty(labelGroupEntityList)) {
            return null;
        }

        for (LabelGroupEntity labelGroupEntity : labelGroupEntityList) {
            GroupLabelResponse groupLabelResponse = new GroupLabelResponse();
            groupLabelResponse.setId(labelGroupEntity.getId());
            groupLabelResponse.setLabel(labelGroupEntity.getLabelGroupName());
            groupLabelResponse.setCount(
                    labelMapper.selectCount(
                            new LambdaQueryWrapper<LabelEntity>()
                                    .eq(LabelEntity::getLabelGroupId, labelGroupEntity.getId())
                    )
            );
            list.add(groupLabelResponse);
        }


        return list;
    }


    @Override
    public IPage<GroupLabelResponse> selectGroupLabelPage(Page<GroupLabelResponse> pageParam) {
        return labelGroupMapper.selectGroupLabelPage(pageParam);


    }


    @Override
    public int addDataSetAndLabel(AddDataSetLabelRequest request) {
        List<LabelEntity> labelEntityList = labelMapper.selectList(
                new LambdaQueryWrapper<LabelEntity>()
                        .eq(LabelEntity::getLabelGroupId, request.getLabelGroupId())
        );
        if (CollectionUtils.isEmpty(labelEntityList)) {
            return 1;
            //  throw new RuntimeException("标签组不存在，新增失败！");
        }
        // 查出之前的标签组
        List<DataSonLabelEntity> list = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, request.getSonId()));

        if (!CollectionUtils.isEmpty(list)) {
            List<Integer> ids = new ArrayList<>();
            for (DataSonLabelEntity dataSonLabelEntity : list) {
                LabelEntity entity = labelMapper.selectById(dataSonLabelEntity.getLabelId());
                if (!ObjectUtils.isEmpty(entity) && entity.getLabelGroupId() != 0) {
                    // id
                    ids.add(dataSonLabelEntity.getId());
                }
            }
            if (!CollectionUtils.isEmpty(ids)) {
                // 进行批量删除
                dataSonLabelMapper.deleteBatchIds(ids);
            }

        }
        for (LabelEntity labelEntity : labelEntityList) {
            DataSonLabelEntity dataSonLabelEntity = dataSonLabelMapper.selectOne(
                    new LambdaQueryWrapper<DataSonLabelEntity>()
                            .eq(DataSonLabelEntity::getLabelId, labelEntity.getId())
                            .eq(DataSonLabelEntity::getSonId, request.getSonId())
            );

            if (ObjectUtils.isEmpty(dataSonLabelEntity)) {
                // 新增
                DataSonLabelEntity entity = new DataSonLabelEntity();
                entity.setSonId(request.getSonId());
                entity.setLabelId(labelEntity.getId());


                dataSonLabelMapper.insert(entity);
            }
        }
        return 1;
    }

    @Override
    public List<DataSetLabelResponse> selectDataSetLabel(String sonId) {
        return labelMapper.selectDataSetLabel(sonId);
    }

    @Override
    public IPage<DataSetLabelResponse> selectDataSetLabelPage(Page<DataSetLabelResponse> pageParam, String sonId, String labelName) {
        if (StringUtils.isEmpty(sonId)) {
            return new Page<>(0, 0);
        }
        return labelGroupMapper.selectDataSetLabelPage(pageParam, sonId, labelName);
    }


    @Override
    public int deleteDataSetLabel(DeleteDataSetLabelRequest request) {
        return dataSonLabelMapper.delete(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, request.getSonId())
                        .eq(DataSonLabelEntity::getLabelId, request.getLabelId())
        );
    }

    @Override
    public int addSaveLabel(LabelEntityRequest label) {
        label.setLabelGroupId(0);
        label.setCreateTime(new Date());
        int result = labelMapper.insert(label);

        if (result < SystemConstant.MAX_SIZE) {
            throw new RuntimeException("标签新增失败！");
        }

        DataSonLabelEntity entity = new DataSonLabelEntity();
        entity.setSonId(label.getSonId());
        entity.setLabelId(label.getId());
        entity.setLabelCount(0);
        dataSonLabelMapper.insert(entity);

        return result;

    }

    @Override
    public int updateDataLabel(LabelEntityRequest label) {
        try {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, label.getSonId())
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("数据集对象不存在！");
            }
            if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                String[] fileIds = dataSonEntity.getFileIds().split(",");
                for (int i = 0; i < fileIds.length; i++) {
                    MarkInfoEntity markInfoEntity = markInfoMapper.selectOne(
                            new LambdaQueryWrapper<MarkInfoEntity>()
                                    .eq(MarkInfoEntity::getFileId, fileIds[i])
                    );
                    // 判断该图片有无标注信息
                    if (ObjectUtils.isEmpty(markInfoEntity)) {  //无标注信息
                        continue;
                    }
                    if (StringUtils.isEmpty(markInfoEntity.getMarkInfo())) {
                        throw new RuntimeException("标注信息为空！");
                    }
                    JsonElement json = new JsonParser().parse(markInfoEntity.getMarkInfo());
                    Map map = new HashMap();
                    map.put("textId", "小浩同学");
                    System.out.println(JsonUtil.replaceJsonNode(json, map).toString());

                  /*  // 准备解析json，替换标签
                    JSONArray json = (JSONArray) JSONArray.parse(markInfoEntity.getMarkInfo());
                    for (int j = 0; j < json.size(); j++) {
                        System.out.println(json.getJSONObject(j).get("props"));
                        JsonElement json = new JsonParser().parse(markInfoEntity.getMarkInfo());

                    }*/

                }


            }


        } catch (RuntimeException e) {
            throw new RuntimeException(e);
        }


        return 0;
    }


    @Override
    public boolean isisExistLabelonlyId(Integer groupId, String onlyId, Integer labelId) {
        LabelGroupEntity labelGroupEntity = labelGroupMapper.selectById(groupId);
        if (ObjectUtils.isEmpty(labelGroupEntity)) {
            return true;
        }
        LambdaQueryWrapper<LabelEntity> queryWrapper = new LambdaQueryWrapper<LabelEntity>()
                .eq(LabelEntity::getLabelGroupId, groupId)
                .eq(LabelEntity::getOnlyId, onlyId.trim());
        if (labelId != null) {
            queryWrapper.ne(LabelEntity::getId, labelId);
        }
        Integer count = labelMapper.selectCount(queryWrapper);
        return count != 0;
    }


    @Override
    public boolean isExistLabelName(Integer groupId, String labelName, Integer labelId) {
        LabelGroupEntity labelGroupEntity = labelGroupMapper.selectById(groupId);
        if (ObjectUtils.isEmpty(labelGroupEntity)) {
            return true;
        }

        LambdaQueryWrapper<LabelEntity> queryWrapper = new LambdaQueryWrapper<LabelEntity>()
                .eq(LabelEntity::getLabelGroupId, groupId)
                .eq(LabelEntity::getLabelName, labelName.trim());

        if (labelId != null) {
            queryWrapper.ne(LabelEntity::getId, labelId);
        }

        Integer count = labelMapper.selectCount(queryWrapper);
        return count != 0;
    }

    @Override
    public boolean isExistLabelName1(Integer labelId, String sonId, String labelName) {

        LambdaQueryWrapper<DataSonLabelEntity> wrapper = new LambdaQueryWrapper<DataSonLabelEntity>()
                .eq(DataSonLabelEntity::getSonId, sonId);

        if (labelId != null) {
            wrapper.ne(DataSonLabelEntity::getLabelId, labelId);
        }

        List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(wrapper);
        if (ObjectUtils.isEmpty(dataSonLabelEntityList)) {
            return false;
        }
        List<Integer> labelIds
                = dataSonLabelEntityList.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());

        List<LabelEntity> labelEntityList = labelMapper.selectListAll(labelIds);
        if (ObjectUtils.isEmpty(labelEntityList)) {
            return false;
        }

        // 第一种情况
        for (LabelEntity labelEntity : labelEntityList) {
            if (labelName.trim().equals(labelEntity.getLabelName().trim())) {
                return true;
            }
        }

        return false;
    }


    @Override
    public boolean isExistEnglishLabelName(Integer groupId, String englishLabelName, Integer labelId, LabelGroupEntity labelGroup) {
        if (ObjectUtils.isEmpty(labelGroup)) {
            return true;
        }

        if (!englishLabelName.startsWith(labelGroup.getEnglishLabelGroupName())) {
            englishLabelName = labelGroup.getEnglishLabelGroupName() + "_" + englishLabelName;
        }

        LambdaQueryWrapper<LabelEntity> queryWrapper = new LambdaQueryWrapper<LabelEntity>()
                .eq(LabelEntity::getLabelGroupId, groupId)
                .eq(LabelEntity::getEnglishLabelName, englishLabelName);
        if (labelId != null) {
            queryWrapper.ne(LabelEntity::getId, labelId);
        }
        Integer count = labelMapper.selectCount(queryWrapper);
        return count != 0;
    }


    @Override
    public boolean isExistEnglishLabelName1(Integer labelId, String sonId, String englishLabelName) {

        LambdaQueryWrapper<DataSonLabelEntity> wrapper = new LambdaQueryWrapper<DataSonLabelEntity>()
                .eq(DataSonLabelEntity::getSonId, sonId);

        if (labelId != null) {
            wrapper.ne(DataSonLabelEntity::getLabelId, labelId);
        }

        List<DataSonLabelEntity> dataSonLabelEntityList = dataSonLabelMapper.selectList(wrapper);
        if (ObjectUtils.isEmpty(dataSonLabelEntityList)) {
            return false;
        }
        List<Integer> labelIds
                = dataSonLabelEntityList.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());

        List<LabelEntity> labelEntityList = labelMapper.selectListAll(labelIds);
        if (ObjectUtils.isEmpty(labelEntityList)) {
            return false;
        }

        // 第一种情况
        for (LabelEntity labelEntity : labelEntityList) {
            if (englishLabelName.trim().equals(labelEntity.getEnglishLabelName().trim())) {
                return true;
            }
        }

        return false;
    }

    @Override
    public int topUpLabel(Integer labelId, String sonId) {
        LabelEntity labelEntity = labelMapper.selectById(labelId);
        int labelMaxSort = getLabelMaxSort(sonId);
        labelEntity.setLabelSort(labelMaxSort + 1);
        System.out.println("当前最大标签值：" + labelMaxSort);
        return labelMapper.updateById(labelEntity);
    }

    @Override
    public int deleteDataSetLabelRequest(DeleteDataSetLabelRequest deleteDataSetLabelRequest) {
        // 查询数据集是否存在
        DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(deleteDataSetLabelRequest.getSonId());
        if (dataSonEntity == null) {
            return 0;
        }

        // 检查标签ID是否为空
        if (StringUtils.isEmpty(deleteDataSetLabelRequest.getLabelIds())) {
            return 0;
        }

        // 查出之前的数据集绑定的所有标签
        List<DataSonLabelEntity> oldLabelList = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, dataSonEntity.getSonId())
        );

        // 提取已存在的标签ID，用于后续判断
        Set<Integer> existingLabelIds = new HashSet<>();
        for (DataSonLabelEntity oldLabel : oldLabelList) {
            existingLabelIds.add(oldLabel.getLabelId());
        }

        // 处理新标签，只添加不存在的标签
        List<DataSonLabelEntity> dataSonLabelEntityList = new ArrayList<>();
        for (String labelIdStr : deleteDataSetLabelRequest.getLabelIds().split(",")) {
            try {
                int labelId = Integer.parseInt(labelIdStr);
                // 检查标签是否已存在，不存在才添加
                if (!existingLabelIds.contains(labelId)) {
                    DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
                    dataSonLabelEntity.setSonId(dataSonEntity.getSonId());
                    dataSonLabelEntity.setLabelId(labelId);
                    dataSonLabelEntityList.add(dataSonLabelEntity);
                    // 将已添加的标签ID加入集合，避免重复添加同一个新标签
                    existingLabelIds.add(labelId);
                }
            } catch (NumberFormatException e) {
                // 处理无效的数字格式
                System.out.println("标签新增失败：" + e.getMessage());
                // 可以选择跳过或抛出异常，根据业务需求决定
            }
        }

        // 只有当有新标签需要添加时才执行批量插入
        if (!dataSonLabelEntityList.isEmpty()) {
            dataSonLabelMapper.insertBatch(dataSonLabelEntityList);
            return !dataSonLabelEntityList.isEmpty() ? 1 : 0; // 返回实际添加的标签数量
        }

        return 1; // 没有添加任何标签
    }


    // 取出当前标签排序的最大值
    public int getLabelMaxSort(String sonId) {
        if (StringUtils.isEmpty(sonId)) {
            return 0;
        }

        List<Integer> labelIds = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, sonId)
        ).stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());


        List<Integer> list = labelMapper.selectBatchIds(labelIds)
                .stream()
                //.filter(LabelEntity -> LabelEntity.getLabelGroupId() != null) // 过滤 groupId 为 null 的数据
                .map(LabelEntity::getLabelSort)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        if (CollectionUtils.isEmpty(list)) {
            return 0;
        }

        // 查找最大的 LabelSort 值
        Optional<Integer> maxLabelSort = list.stream().max(Comparator.naturalOrder());
        return maxLabelSort.orElse(0);
    }


}
