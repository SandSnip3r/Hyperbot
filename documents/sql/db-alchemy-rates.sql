SELECT
    r.ID, 
    r.CodeName128, 
    CASE WHEN TypeID4 = 1 THEN 'Elixir' ELSE 'Powder' END as 'ItemType', 
    CASE WHEN TypeID4 = 1 THEN CASE WHEN i.ItemClass = 1 THEN 'A' ELSE 'B' END ELSE 'D' + CONVERT(varchar(25), i.ItemClass) END as 'Type',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param2), 1, 1))) + '%' as '+1',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param2), 2, 1))) + '%' as '+2',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param2), 3, 1))) + '%' as '+3',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param2), 4, 1))) + '%' as '+4',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param3), 1, 1))) + '%' as '+5',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param3), 2, 1))) + '%' as '+6',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param3), 3, 1))) + '%' as '+7',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param3), 4, 1))) + '%' as '+8',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param4), 1, 1))) + '%' as '+9',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param4), 2, 1))) + '%' as '+10',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param4), 3, 1))) + '%' as '+11',
    CONVERT(varchar(25), CONVERT(INT, SUBSTRING(CONVERT(varbinary(4), i.Param4), 4, 1))) + '%' as '+12'
FROM    dbo._RefObjCommon r with(NOLOCK)
JOIN    dbo._RefObjitem i with(NOLOCK) on r.Link = i.ID
WHERE    TypeID1 = 3 AND TypeID2 = 3 AND TypeID3 = 10 AND TypeID4 in (1,2)

ORDER BY ItemType asc, i.ItemClass asc