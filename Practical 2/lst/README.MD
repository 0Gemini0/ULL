The eval script is:
#python lst_gap.py lst_test.gold lst.out out no-mwe

lst_test.gold   gold standar annotation 

lst.out         prediction of system 

out             output log of evaluation script

no-mwe          do not count multi-word-expressions


The last.out format of your system is tab separated:
#RANKED target word sentence id sub score sub score ...
RANKED| side.n 301| hand -2.7787|   part -5.9708|   against -7.5731|shore -11.5640| surface -11.5733|   faction -12.7584|   facet -13.6011| aspect -13.6443|bank -14.4334|  perspective -1     5.0994|   area -15.1572|  standpoint -15.1597|view -15.4184|  divide -16.0836|conversely -16.7894|ally -18.5698|  edge -19.0853|  contingent -19.3950|team -21.0785|  boundary -21.1545|       fringe -22.5431|flank -24.0384| position -32.6632|  instead -33.1494|   you -35.0130|   responsibility -35.7164

The file used as input for a system is lst_test.preprocessed
with format:

#target word, id_sentence,    position of target word in sentence,    sentence

side.n	301	29	on sunday at craven cottage , jose mourinho and his all stars exhibited all of the above symptoms and they were made to pay the price by a fulham side that had in previous weeks woken up after matches with their heads kicked in .

The gold standard substitution list is: lst.gold.candidates

#target word:: sub1; sub2; ... subN
scrap.n::discarded item;album;clipping;piece;crumb;fragment;recycling;morsel;collage;shred;odds and ends;rubbish;waste metal;leftover;bit;waste;remainder;trade in
