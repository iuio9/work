package com.qczy.service.impl;

import cn.hutool.core.convert.Convert;
import cn.hutool.core.lang.Dict;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.baomidou.mybatisplus.core.toolkit.ObjectUtils;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.DataFatherMapper;
import com.qczy.mapper.DataSonMapper;
import com.qczy.mapper.DictDataMapper;
import com.qczy.mapper.DictTypeMapper;
import com.qczy.model.entity.DataFatherEntity;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.entity.MenuEntity;
import com.qczy.model.response.DictDataTreeResponse;
import com.qczy.model.response.DictSetTypeResponse;
import com.qczy.service.DictDataService;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 9:53
 * @Description:
 */
@Service
public class DictDataServiceImpl extends ServiceImpl<DictDataMapper, DictDataEntity> implements DictDataService {


    @Autowired
    private DictDataMapper dictDataMapper;

    @Autowired
    private DataFatherMapper dataFatherMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public DictDataEntity selectDictDataById(Integer id) {
        return dictDataMapper.selectDictDataById(id);
    }

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息
     * @param typeId    类型id
     * @param dictData  查询类型
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<DictDataEntity> selectDictDataList(Page<DictDataEntity> pageParam, Integer typeId, DictDataEntity dictData) {
        return dictDataMapper.selectDictDataList(pageParam, typeId, dictData);
    }

    /**
     * 新增【请填写功能名称】
     *
     * @param dictData 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertDictData(DictDataEntity dictData) {
        dictData.setCreateTime(new Date());
        return dictDataMapper.insertDictData(dictData);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param dictData 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateDictData(DictDataEntity dictData) {
        dictData.setUpdateTime(new Date());
        return dictDataMapper.updateDictData(dictData);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteDictDataByIds(int[] ids) {
        return dictDataMapper.deleteDictDataByIds(ids);
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteDictDataById(Integer id) {
        return dictDataMapper.deleteDictDataById(id);
    }

    @Override
    public List<DictDataEntity> getDictDataTree(Integer typeId) {
        List<DictDataEntity> dictDataSelect = null;
        List<DictDataEntity> dictDataList = dictDataMapper.selectList(new LambdaQueryWrapper<DictDataEntity>().eq(DictDataEntity::getTypeId, typeId));
        if (!CollectionUtils.isEmpty(dictDataList)) {
            dictDataSelect = getDataSelect(dictDataList);
        }


        return dictDataSelect;
    }

    /**
     * 获取数据集字典树形结构，包含数量统计
     */
    public List<DictSetTypeResponse> selectDataSetDictList() {
        // 查询所有字典数据
        List<DictSetTypeResponse> allNodes = dictDataMapper.selectDataSetDictList();
        if (CollectionUtils.isEmpty(allNodes)) {
            return new ArrayList<>();
        }

        // 初始化每个节点的基础数量
        initNodeCounts(allNodes);

        // 构建树形结构并计算总数量
        return buildTreeStructure(allNodes);
    }

    @Override
    public List<DictDataEntity> selectBatchId(Integer[] ids) {
        return dictDataMapper.selectBatchIds(Arrays.asList(ids));
    }

    @Override
    public String getTreeLevelDict(Integer dataTypeId) {
        DictDataEntity dictDataEntity = dictDataMapper.selectById(dataTypeId);
        if (ObjectUtils.isEmpty(dictDataEntity)) {
            throw new RuntimeException("对象不存在！");
        }
        if (dictDataEntity.getParentId() == 0) {
            return dictDataEntity.getDictLabel();
        }
        // 存在多级
        // 记录label
        StringBuilder sb = new StringBuilder();
        sb.insert(0, dictDataEntity.getDictLabel() + "-");
        Integer parentId = dictDataEntity.getParentId();
        while (true) {
            DictDataEntity dictDataEntity1 = dictDataMapper.selectById(parentId);
            if (dictDataEntity1.getParentId() != 0) {
                if (ObjectUtils.isEmpty(dictDataEntity1)) {
                    continue;
                }
                sb.insert(0, dictDataEntity1.getDictLabel() + "-");
                parentId = dictDataEntity1.getParentId();
            } else {
                sb.insert(0, dictDataEntity1.getDictLabel() + "-");
                return sb.deleteCharAt(sb.length() - 1).toString();
            }

        }
    }


    @Override
    public List<Integer> getTreeLevelDictIds(Integer dataTypeId) {
        DictDataEntity dictDataEntity = dictDataMapper.selectById(dataTypeId);
        if (ObjectUtils.isEmpty(dictDataEntity)) {
            throw new RuntimeException("对象不存在！");
        }
        List<Integer> ids = new ArrayList<>();
        if (dictDataEntity.getParentId() == 0) {
            ids.add(dictDataEntity.getId());
            return ids;
        }
        // 存在多级
        // 记录label
        Integer parentId = dictDataEntity.getParentId();
        while (true) {
            DictDataEntity dictDataEntity1 = dictDataMapper.selectById(parentId);
            if (dictDataEntity1.getParentId() != 0) {
                if (ObjectUtils.isEmpty(dictDataEntity1)) {
                    continue;
                }
                ids.add(0, dictDataEntity1.getId());
                parentId = dictDataEntity1.getParentId();
            } else {
                ids.add(0, dictDataEntity1.getId());
                return ids;
            }
        }
    }


    @Override
    public int getFatherAndDataSon(int[] ids) {
        return dataFatherMapper.selectCount(
                new LambdaQueryWrapper<DataFatherEntity>()
                        .in(DataFatherEntity::getDataTypeId, StringUtils.strip(Arrays.toString(ids), "[]"))
        );
    }


    public List<DictDataEntity> getDataSelect(List<DictDataEntity> dictDataList) {
        List<DictDataEntity> dictDataEntities = new ArrayList<>();
        //判断集合是否为空，如果为空，直接返回
        if (CollectionUtils.isEmpty(dictDataList)) {
            return null;
        }

        //根节点为0,找下面的子节点
        for (DictDataEntity dictDataEntity : dictDataList) {
            if (dictDataEntity.getParentId() == 0) {
                dictDataEntities.add(children(dictDataEntity, dictDataList));
            }
        }
        return dictDataEntities;
    }


    //执行递归操作
    private static DictDataEntity children(DictDataEntity dictData, List<DictDataEntity> dataList) {
        dictData.setChildren(new ArrayList<DictDataEntity>());
        for (DictDataEntity data : dataList) {
            //拿着父节点的id去查询所有的子节点       id == parentId
            if (dictData.getId().equals(data.getParentId())) {
                if (dictData.getChildren() == null) {
                    dictData.setChildren(new ArrayList<DictDataEntity>());
                }
                //以此循环执行，直到找不到子节点结束...
                dictData.getChildren().add(children(data, dataList));
            }
        }
        return dictData;
    }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------


    /**
     * 初始化每个节点的当前节点数量（nodeSumCount）
     */
    private void initNodeCounts(List<DictSetTypeResponse> nodes) {
        for (DictSetTypeResponse node : nodes) {
            // 查询当前节点自身的文件数量（不包含子节点）
            Long currentFileCount = dataSonMapper.countCurrentNodeFiles(Long.valueOf(node.getId()));
            node.setNodeSumCount(currentFileCount != null ? currentFileCount : 0L);
        }
    }

    /**
     * 构建树形结构并计算总数量（fileSumCount）
     */
    private List<DictSetTypeResponse> buildTreeStructure(List<DictSetTypeResponse> allNodes) {
        List<DictSetTypeResponse> rootNodes = new ArrayList<>();

        // 筛选根节点（parentId为0）
        for (DictSetTypeResponse node : allNodes) {
            if (node.getParentId() == 0) {
                // 递归计算子节点并汇总总数量
                DictSetTypeResponse root = buildChildren(node, allNodes);
                rootNodes.add(root);
            }
        }

        return rootNodes;
    }

    /**
     * 递归构建子节点并计算总数量
     */
    private DictSetTypeResponse buildChildren(DictSetTypeResponse currentNode, List<DictSetTypeResponse> allNodes) {
        // 初始化子节点列表
        currentNode.setChildren(new ArrayList<>());

        // 总数量初始值 = 当前节点自身数量
        Long totalCount = currentNode.getNodeSumCount();

        // 查找所有子节点
        for (DictSetTypeResponse node : allNodes) {
            if (currentNode.getId().equals(node.getParentId())) {
                // 递归处理子节点
                DictSetTypeResponse childNode = buildChildren(node, allNodes);
                currentNode.getChildren().add(childNode);

                // 累加子节点的总数量
                totalCount += childNode.getFileSumCount();
            }
        }

        // 设置当前节点的总数量（自身 + 所有子节点总和）
        currentNode.setFileSumCount(totalCount);
        return currentNode;
    }


    public List<DictSetTypeResponse> getDataSelectTwo(List<DictSetTypeResponse> dictDataList) {
        List<DictSetTypeResponse> dictDataEntities = new ArrayList<>();
        //判断集合是否为空，如果为空，直接返回
        if (CollectionUtils.isEmpty(dictDataList)) {
            return null;
        }

        //根节点为0,找下面的子节点
        for (DictSetTypeResponse dictDataEntity : dictDataList) {
            if (dictDataEntity.getParentId() == 0) {
                dictDataEntities.add(childrenTwo(dictDataEntity, dictDataList));
            }
        }
        return dictDataEntities;
    }

    //执行递归操作
    private static DictSetTypeResponse childrenTwo(DictSetTypeResponse dictData, List<DictSetTypeResponse> dataList) {
        dictData.setChildren(new ArrayList<DictSetTypeResponse>());
        for (DictSetTypeResponse data : dataList) {
            //拿着父节点的id去查询所有的子节点       id == parentId
            if (dictData.getId().equals(data.getParentId())) {
                if (dictData.getChildren() == null) {
                    dictData.setChildren(new ArrayList<DictSetTypeResponse>());
                    //data.setFileNumber();
                }
                //以此循环执行，直到找不到子节点结束...
                dictData.getChildren().add(childrenTwo(data, dataList));
            }
        }
        return dictData;
    }
}
